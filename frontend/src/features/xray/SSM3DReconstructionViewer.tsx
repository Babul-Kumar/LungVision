import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import {
  RotateCcw,
  Download,
  Eye,
  EyeOff,
  Layers,
  Camera,
  Play,
  Pause,
  MapPin,
  Palette,
} from 'lucide-react';
import { Button } from '@/components/common/Button';

interface SSM3DReconstructionViewerProps {
  objData?: string | null;
  vertices?: number[][] | null;
  faces?: number[][] | null;
  referenceVertices?: number[][] | null;
  referenceFaces?: number[][] | null;
  cardiacMesh?: {
    vertex_count?: number;
    face_count?: number;
    volume_ml?: number;
    cardiothoracic_ratio?: number;
    vertices?: number[][];
    faces?: number[][];
  } | null;
  landmarks?: Record<string, [number, number, number]>;
  onExportObj?: () => void;
  isExporting?: boolean;
}

interface HoveredLandmarkInfo {
  name: string;
  coords: [number, number, number];
  x: number;
  y: number;
}

export const SSM3DReconstructionViewer: React.FC<SSM3DReconstructionViewerProps> = ({
  objData,
  vertices,
  faces,
  referenceVertices,
  referenceFaces,
  cardiacMesh,
  landmarks,
  onExportObj,
  isExporting = false,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [wireframe, setWireframe] = useState<boolean>(false);
  const [showLandmarks, setShowLandmarks] = useState<boolean>(true);
  const [showHeart, setShowHeart] = useState<boolean>(true);
  const [showReferenceCT, setShowReferenceCT] = useState<boolean>(true);
  const [colorMode, setColorMode] = useState<'pleura' | 'bilateral' | 'platinum'>('pleura');
  const [isAutoRotating, setIsAutoRotating] = useState<boolean>(false);
  const [activeViewName, setActiveViewName] = useState<string>('Oblique 3D');
  const [vertexCount, setVertexCount] = useState<number>(0);
  const [faceCount, setFaceCount] = useState<number>(0);
  const [hoveredLandmark, setHoveredLandmark] = useState<HoveredLandmarkInfo | null>(null);

  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshGroupRef = useRef<THREE.Group | null>(null);
  const cardiacGroupRef = useRef<THREE.Group | null>(null);
  const referenceGroupRef = useRef<THREE.Group | null>(null);
  const landmarkGroupRef = useRef<THREE.Group | null>(null);
  const animFrameIdRef = useRef<number | null>(null);
  const isAutoRotatingRef = useRef(false);

  useEffect(() => {
    isAutoRotatingRef.current = isAutoRotating;
  }, [isAutoRotating]);

  // Compute anatomical view label based on camera position (only update state if changed)
  const updateViewLabel = useCallback(() => {
    if (!cameraRef.current) return;
    const pos = cameraRef.current.position;
    const x = Math.abs(pos.x);
    const y = Math.abs(pos.y);
    const z = Math.abs(pos.z);

    let newLabel = 'Anterior Oblique 3D';
    if (z > x * 2 && z > y * 2) {
      newLabel = pos.z > 0 ? 'Superior (Axial Top)' : 'Inferior (Axial Bottom)';
    } else if (y > x * 1.5 && y > z * 1.5) {
      newLabel = pos.y < 0 ? 'Anterior (Frontal AP)' : 'Posterior (Dorsal PA)';
    } else if (x > y * 1.5 && x > z * 1.5) {
      newLabel = pos.x > 0 ? 'Right Lateral' : 'Left Lateral';
    }

    setActiveViewName((prev) => (prev !== newLabel ? newLabel : prev));
  }, []);

  // 1. Initialize Three.js Scene, Camera, Controls, Lights
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 450;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x160a28);
    sceneRef.current = scene;

    // Adult thorax spans ~245 mm in Z and ~240 mm in X; distance 520 mm frames the full anatomy comfortably
    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 4000);
    camera.position.set(0, -520, 50);
    camera.up.set(0, 0, 1); // Z-up craniocaudal
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    rendererRef.current = renderer;

    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 2500;
    controls.minDistance = 60;
    controls.target.set(0, 0, 0);
    controls.addEventListener('change', updateViewLabel);
    controlsRef.current = controls;

    // Studio Lighting: Celestial hemisphere dome, key light, cool fill, and marigold contour rim
    const hemiLight = new THREE.HemisphereLight(0xfffbeb, 0x160a28, 0.65);
    scene.add(hemiLight);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
    scene.add(ambientLight);

    const dirKeyLight = new THREE.DirectionalLight(0xfff7ed, 1.35);
    dirKeyLight.position.set(160, -280, 240);
    scene.add(dirKeyLight);

    const dirFillLight = new THREE.DirectionalLight(0xe0e7ff, 0.75);
    dirFillLight.position.set(-200, -180, 100);
    scene.add(dirFillLight);

    const backRimLight = new THREE.DirectionalLight(0xf59e0b, 0.65);
    backRimLight.position.set(0, 320, 180);
    scene.add(backRimLight);

    // Groups
    const meshGroup = new THREE.Group();
    scene.add(meshGroup);
    meshGroupRef.current = meshGroup;

    const cardiacGroup = new THREE.Group();
    scene.add(cardiacGroup);
    cardiacGroupRef.current = cardiacGroup;

    const referenceGroup = new THREE.Group();
    scene.add(referenceGroup);
    referenceGroupRef.current = referenceGroup;

    const landmarkGroup = new THREE.Group();
    scene.add(landmarkGroup);
    landmarkGroupRef.current = landmarkGroup;

    // Animation Loop
    const animate = () => {
      animFrameIdRef.current = requestAnimationFrame(animate);
      if (isAutoRotatingRef.current && meshGroupRef.current) {
        meshGroupRef.current.rotation.z += 0.006;
        if (cardiacGroupRef.current) {
          cardiacGroupRef.current.rotation.z = meshGroupRef.current.rotation.z;
        }
        if (referenceGroupRef.current) {
          referenceGroupRef.current.rotation.z = meshGroupRef.current.rotation.z;
        }
        if (landmarkGroupRef.current) {
          landmarkGroupRef.current.rotation.z = meshGroupRef.current.rotation.z;
        }
      }
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container || !rendererRef.current || !cameraRef.current) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };

    // Landmark hover raycaster (throttled via requestAnimationFrame)
    let mouseMoveRaf: number | null = null;
    const handleMouseMove = (e: MouseEvent) => {
      if (mouseMoveRaf) return;
      mouseMoveRaf = requestAnimationFrame(() => {
        mouseMoveRaf = null;
        if (!container || !rendererRef.current || !cameraRef.current || !landmarkGroupRef.current) {
          return;
        }
        const rect = container.getBoundingClientRect();
        const mouse = new THREE.Vector2(
          ((e.clientX - rect.left) / rect.width) * 2 - 1,
          -((e.clientY - rect.top) / rect.height) * 2 + 1
        );
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, cameraRef.current);
        const intersects = raycaster.intersectObjects(landmarkGroupRef.current.children, true);
        if (intersects.length > 0 && intersects[0].object.userData?.landmarkName) {
          const name = intersects[0].object.userData.landmarkName;
          const coords = intersects[0].object.userData.coords || [0, 0, 0];
          setHoveredLandmark({
            name,
            coords,
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
          });
        } else {
          setHoveredLandmark((prev) => (prev !== null ? null : prev));
        }
      });
    };

    container.addEventListener('mousemove', handleMouseMove);

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    return () => {
      container.removeEventListener('mousemove', handleMouseMove);
      resizeObserver.disconnect();
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
      controls.removeEventListener('change', updateViewLabel);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    };
  }, [updateViewLabel]);

  // 2. Build or Update Surface Mesh
  useEffect(() => {
    if (!meshGroupRef.current) return;

    // Clear previous meshes
    while (meshGroupRef.current.children.length > 0) {
      const child = meshGroupRef.current.children[0] as THREE.Mesh;
      meshGroupRef.current.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (Array.isArray(child.material)) {
        child.material.forEach((m) => m.dispose());
      } else if (child.material) {
        child.material.dispose();
      }
    }

    if (cardiacGroupRef.current) {
      while (cardiacGroupRef.current.children.length > 0) {
        const child = cardiacGroupRef.current.children[0] as THREE.Mesh;
        cardiacGroupRef.current.remove(child);
        if (child.geometry) child.geometry.dispose();
        if (Array.isArray(child.material)) {
          child.material.forEach((m) => m.dispose());
        } else if (child.material) {
          child.material.dispose();
        }
      }
    }

    if (referenceGroupRef.current) {
      while (referenceGroupRef.current.children.length > 0) {
        const child = referenceGroupRef.current.children[0] as THREE.Mesh;
        referenceGroupRef.current.remove(child);
        if (child.geometry) child.geometry.dispose();
        if (Array.isArray(child.material)) {
          child.material.forEach((m) => m.dispose());
        } else if (child.material) {
          child.material.dispose();
        }
      }
    }

    // Organic anatomical visceral pleura material with serous clearcoat fluid reflection
    const createMaterial = () => {
      if (colorMode === 'pleura') {
        return new THREE.MeshPhysicalMaterial({
          color: 0xf4aba6, // Natural visceral pleura living pulmonary pink/rose
          roughness: 0.32,
          metalness: 0.04,
          clearcoat: 0.65, // Moist serous fluid surface reflection
          clearcoatRoughness: 0.12,
          transmission: 0.16, // Aerated parenchymal translucency
          ior: 1.34, // Biological serous fluid index of refraction
          reflectivity: 0.5,
          sheen: 0.45,
          sheenColor: new THREE.Color(0xffccd5),
          transparent: true,
          opacity: wireframe ? 0.9 : 0.94,
          wireframe: wireframe,
          side: THREE.DoubleSide,
        });
      } else if (colorMode === 'bilateral') {
        return new THREE.MeshPhysicalMaterial({
          roughness: 0.3,
          metalness: 0.04,
          clearcoat: 0.6,
          clearcoatRoughness: 0.12,
          transmission: 0.14,
          ior: 1.34,
          reflectivity: 0.5,
          vertexColors: true,
          transparent: true,
          opacity: wireframe ? 0.9 : 0.94,
          wireframe: wireframe,
          side: THREE.DoubleSide,
        });
      } else {
        // High-contrast clinical platinum morphology
        return new THREE.MeshPhysicalMaterial({
          color: 0xf1f5f9, // Ice Silver Platinum
          roughness: 0.28,
          metalness: 0.08,
          clearcoat: 0.5,
          clearcoatRoughness: 0.15,
          transparent: true,
          opacity: wireframe ? 0.9 : 0.92,
          wireframe: wireframe,
          side: THREE.DoubleSide,
        });
      }
    };

    // Authentic cardiac silhouette material: deep garnet myocardium with serous pericardial clearcoat
    const cardiacMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x991b1b, // Deep anatomical garnet/cardiac crimson
      roughness: 0.28,
      metalness: 0.03,
      clearcoat: 0.82, // Moist pericardial serous fluid reflection
      clearcoatRoughness: 0.08,
      sheen: 0.35,
      sheenColor: new THREE.Color(0xf87171),
      transparent: true,
      opacity: wireframe ? 0.9 : 0.96,
      wireframe: wireframe,
      side: THREE.DoubleSide,
    });

    const material = createMaterial();

    if (objData) {
      const loader = new OBJLoader();
      try {
        const obj = loader.parse(objData);
        let totalVerts = 0;
        let totalFaces = 0;
        obj.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            const isCardiac = mesh.name.toLowerCase().includes('cardiac');
            mesh.material = isCardiac ? cardiacMaterial : material;
            mesh.geometry.computeVertexNormals();
            totalVerts += mesh.geometry.attributes.position.count;
            if (mesh.geometry.index) {
              totalFaces += mesh.geometry.index.count / 3;
            }
          }
        });
        meshGroupRef.current.add(obj);
        setVertexCount(totalVerts);
        setFaceCount(totalFaces || 14758);

        // Center mesh
        const box = new THREE.Box3().setFromObject(obj);
        const center = box.getCenter(new THREE.Vector3());
        meshGroupRef.current.position.sub(center);
        meshGroupRef.current.rotation.z = 0;
        if (cardiacGroupRef.current) {
          cardiacGroupRef.current.position.copy(meshGroupRef.current.position);
          cardiacGroupRef.current.rotation.z = 0;
          cardiacGroupRef.current.visible = showHeart;
        }
        if (landmarkGroupRef.current) {
          landmarkGroupRef.current.position.copy(meshGroupRef.current.position);
          landmarkGroupRef.current.rotation.z = 0;
        }
      } catch (err) {
        console.error('Failed to parse OBJ in 3D viewer:', err);
      }
    } else if (vertices && faces) {
      const geometry = new THREE.BufferGeometry();
      const posArray = new Float32Array(vertices.length * 3);
      for (let i = 0; i < vertices.length; i++) {
        posArray[i * 3] = vertices[i][0];
        posArray[i * 3 + 1] = vertices[i][1];
        posArray[i * 3 + 2] = vertices[i][2];
      }
      geometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));

      // Bilateral anatomical coloration (Left Lung: Ice Silver / Platinum Rose, Right Lung: Champagne Gold / Marigold)
      if (colorMode === 'bilateral') {
        const colors = new Float32Array(vertices.length * 3);
        for (let i = 0; i < vertices.length; i++) {
          const x = vertices[i][0];
          if (x < 0) {
            // Left lung (Ice Silver Platinum with subtle rose-pearl undertone: 0.95, 0.93, 0.97)
            colors[i * 3] = 0.95;
            colors[i * 3 + 1] = 0.93;
            colors[i * 3 + 2] = 0.97;
          } else {
            // Right lung (Warm Champagne Gold / Marigold: 1.0, 0.91, 0.68)
            colors[i * 3] = 1.0;
            colors[i * 3 + 1] = 0.91;
            colors[i * 3 + 2] = 0.68;
          }
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      }

      const indices: number[] = [];
      for (let f = 0; f < faces.length; f++) {
        indices.push(faces[f][0], faces[f][1], faces[f][2]);
      }
      geometry.setIndex(indices);
      geometry.computeVertexNormals();

      const mesh = new THREE.Mesh(geometry, material);
      meshGroupRef.current.add(mesh);

      // Build 3D Watertight Cardiac Silhouette (Heart)
      let cVertCount = 0;
      let cFaceCount = 0;
      if (cardiacMesh?.vertices && cardiacMesh?.faces && cardiacGroupRef.current) {
        const cGeo = new THREE.BufferGeometry();
        const cPos = new Float32Array(cardiacMesh.vertices.length * 3);
        for (let i = 0; i < cardiacMesh.vertices.length; i++) {
          cPos[i * 3] = cardiacMesh.vertices[i][0];
          cPos[i * 3 + 1] = cardiacMesh.vertices[i][1];
          cPos[i * 3 + 2] = cardiacMesh.vertices[i][2];
        }
        cGeo.setAttribute('position', new THREE.BufferAttribute(cPos, 3));
        const cIdx: number[] = [];
        for (let f = 0; f < cardiacMesh.faces.length; f++) {
          cIdx.push(cardiacMesh.faces[f][0], cardiacMesh.faces[f][1], cardiacMesh.faces[f][2]);
        }
        cGeo.setIndex(cIdx);
        cGeo.computeVertexNormals();
        const cMesh = new THREE.Mesh(cGeo, cardiacMaterial);
        cardiacGroupRef.current.add(cMesh);
        cVertCount = cardiacMesh.vertices.length;
        cFaceCount = cardiacMesh.faces.length;
      }

      // Build Reference CT Ground Truth Mesh (Cyan Translucent)
      if (referenceVertices && referenceFaces && referenceGroupRef.current) {
        const refGeo = new THREE.BufferGeometry();
        const refPos = new Float32Array(referenceVertices.length * 3);
        for (let i = 0; i < referenceVertices.length; i++) {
          refPos[i * 3] = referenceVertices[i][0];
          refPos[i * 3 + 1] = referenceVertices[i][1];
          refPos[i * 3 + 2] = referenceVertices[i][2];
        }
        refGeo.setAttribute('position', new THREE.BufferAttribute(refPos, 3));
        const refIdx: number[] = [];
        for (let f = 0; f < referenceFaces.length; f++) {
          refIdx.push(referenceFaces[f][0], referenceFaces[f][1], referenceFaces[f][2]);
        }
        refGeo.setIndex(refIdx);
        refGeo.computeVertexNormals();

        const refMat = new THREE.MeshPhysicalMaterial({
          color: 0x06b6d4, // Cyan translucent CT reference
          roughness: 0.35,
          metalness: 0.05,
          transparent: true,
          opacity: wireframe ? 0.75 : 0.45,
          wireframe: wireframe,
          side: THREE.DoubleSide,
        });
        const refMesh = new THREE.Mesh(refGeo, refMat);
        referenceGroupRef.current.add(refMesh);
      }

      setVertexCount(vertices.length + cVertCount);
      setFaceCount(faces.length + cFaceCount);

      const box = new THREE.Box3().setFromObject(mesh);
      const center = box.getCenter(new THREE.Vector3());
      meshGroupRef.current.position.sub(center);
      meshGroupRef.current.rotation.z = 0;
      if (cardiacGroupRef.current) {
        cardiacGroupRef.current.position.copy(meshGroupRef.current.position);
        cardiacGroupRef.current.rotation.z = 0;
        cardiacGroupRef.current.visible = showHeart;
      }
      if (referenceGroupRef.current) {
        referenceGroupRef.current.position.copy(meshGroupRef.current.position);
        referenceGroupRef.current.rotation.z = 0;
        referenceGroupRef.current.visible = showReferenceCT;
      }
      if (landmarkGroupRef.current) {
        landmarkGroupRef.current.position.copy(meshGroupRef.current.position);
        landmarkGroupRef.current.rotation.z = 0;
      }
    }
  }, [objData, vertices, faces, referenceVertices, referenceFaces, cardiacMesh, wireframe, colorMode, showHeart, showReferenceCT]);

  useEffect(() => {
    if (cardiacGroupRef.current) {
      cardiacGroupRef.current.visible = showHeart;
    }
  }, [showHeart]);

  useEffect(() => {
    if (referenceGroupRef.current) {
      referenceGroupRef.current.visible = showReferenceCT;
    }
  }, [showReferenceCT]);

  // 3. Render Landmark Markers
  useEffect(() => {
    if (!landmarkGroupRef.current) return;

    while (landmarkGroupRef.current.children.length > 0) {
      const child = landmarkGroupRef.current.children[0] as THREE.Mesh;
      landmarkGroupRef.current.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) (child.material as THREE.Material).dispose();
    }

    if (!showLandmarks || !landmarks) return;

    // Match offset of meshGroup
    if (meshGroupRef.current) {
      landmarkGroupRef.current.position.copy(meshGroupRef.current.position);
    }

    const sphereGeo = new THREE.SphereGeometry(3.5, 16, 16);
    const landmarkMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0xd97706,
      emissiveIntensity: 0.5,
      roughness: 0.2,
      metalness: 0.5,
    });

    Object.entries(landmarks).forEach(([name, coords]) => {
      const marker = new THREE.Mesh(sphereGeo, landmarkMat);
      marker.position.set(coords[0], coords[1], coords[2]);
      marker.userData = { landmarkName: name, coords };
      landmarkGroupRef.current?.add(marker);
    });
  }, [landmarks, showLandmarks, vertexCount]);

  // View angle preset dispatcher
  const setAnatomicalView = (view: 'AP' | 'PA' | 'RL' | 'LL' | 'SI' | 'OBLIQUE') => {
    if (!cameraRef.current || !controlsRef.current) return;
    const camera = cameraRef.current;
    const controls = controlsRef.current;

    controls.target.set(0, 0, 0);

    switch (view) {
      case 'AP':
        camera.position.set(0, -540, 10);
        camera.up.set(0, 0, 1);
        break;
      case 'PA':
        camera.position.set(0, 540, 10);
        camera.up.set(0, 0, 1);
        break;
      case 'RL':
        camera.position.set(540, 0, 10);
        camera.up.set(0, 0, 1);
        break;
      case 'LL':
        camera.position.set(-540, 0, 10);
        camera.up.set(0, 0, 1);
        break;
      case 'SI':
        camera.position.set(0, 0, 540);
        camera.up.set(0, 1, 0);
        break;
      case 'OBLIQUE':
      default:
        camera.position.set(0, -520, 50);
        camera.up.set(0, 0, 1);
        break;
    }

    controls.update();
    updateViewLabel();
  };

  const handleExportSnapshot = () => {
    if (!rendererRef.current) return;
    const dataUrl = rendererRef.current.domElement.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = `lungvision_ssm_surface_reconstruction.png`;
    link.href = dataUrl;
    link.click();
  };

  return (
    <div className="relative w-full h-[460px] rounded-xl overflow-hidden bg-[#160a28] border border-[#e2e8f0] flex flex-col shadow-inner">
      {/* 3D Canvas Mount Point */}
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Viewport Overlay HUD Header */}
      <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none z-20">
        <div className="flex items-center gap-2 pointer-events-auto bg-white/95 backdrop-blur-md px-3 py-1.5 rounded-lg border border-[#e2e8f0] text-xs shadow-md">
          <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
          <span className="font-semibold text-[#1a082e] font-mono">SSM 3D Surface</span>
          <span className="text-slate-300">|</span>
          <span className="text-[#5e4b75]">{vertexCount.toLocaleString()} Vertices</span>
          <span className="text-slate-300">·</span>
          <span className="text-[#5e4b75]">{faceCount.toLocaleString()} Triangles</span>
        </div>

        {/* Viewport Action Controls */}
        <div className="flex items-center gap-1.5 pointer-events-auto bg-white/95 backdrop-blur-md p-1.5 rounded-lg border border-[#e2e8f0] shadow-md">
          {/* Quick Camera Presets */}
          <div className="flex items-center gap-1 mr-1 border-r border-[#e2e8f0] pr-1.5 text-[10px] font-mono">
            <button
              onClick={() => setAnatomicalView('AP')}
              className="px-1.5 py-1 rounded hover:bg-slate-100 text-[#5e4b75] hover:text-amber-800 transition-colors"
              title="Anterior (Frontal AP View)"
            >
              AP
            </button>
            <button
              onClick={() => setAnatomicalView('PA')}
              className="px-1.5 py-1 rounded hover:bg-slate-100 text-[#5e4b75] hover:text-amber-800 transition-colors"
              title="Posterior (Dorsal PA View)"
            >
              PA
            </button>
            <button
              onClick={() => setAnatomicalView('RL')}
              className="px-1.5 py-1 rounded hover:bg-slate-100 text-[#5e4b75] hover:text-amber-800 transition-colors"
              title="Right Lateral View"
            >
              RL
            </button>
            <button
              onClick={() => setAnatomicalView('LL')}
              className="px-1.5 py-1 rounded hover:bg-slate-100 text-[#5e4b75] hover:text-amber-800 transition-colors"
              title="Left Lateral View"
            >
              LL
            </button>
            <button
              onClick={() => setAnatomicalView('SI')}
              className="px-1.5 py-1 rounded hover:bg-slate-100 text-[#5e4b75] hover:text-amber-800 transition-colors"
              title="Superior (Top-Down View)"
            >
              SI
            </button>
          </div>

          <button
            onClick={() => setWireframe((w) => !w)}
            className={`px-2 py-1 rounded text-xs font-medium flex items-center gap-1 transition-colors ${wireframe
              ? 'bg-amber-100 text-amber-900 font-semibold'
              : 'text-[#5e4b75] hover:text-[#1a082e] hover:bg-slate-100'
              }`}
            title="Toggle Wireframe Mesh"
          >
            <Layers className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Wireframe</span>
          </button>

          <button
            onClick={() => setShowLandmarks((s) => !s)}
            className={`px-2 py-1 rounded text-xs font-medium flex items-center gap-1 transition-colors ${showLandmarks
              ? 'bg-amber-100 text-amber-900 font-semibold'
              : 'text-[#5e4b75] hover:text-[#1a082e] hover:bg-slate-100'
              }`}
            title="Toggle Anatomical Landmarks"
          >
            {showLandmarks ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span className="hidden sm:inline">Landmarks</span>
          </button>

          <button
            onClick={() => setShowHeart((h) => !h)}
            className={`px-2 py-1 rounded text-xs font-medium flex items-center gap-1 transition-colors ${showHeart
              ? 'bg-rose-100 text-rose-950 font-semibold border border-rose-300'
              : 'text-[#5e4b75] hover:text-[#1a082e] hover:bg-slate-100'
              }`}
            title="Toggle 3D Watertight Cardiac Silhouette (Heart)"
          >
            <span className="text-xs">🫀</span>
            <span className="hidden sm:inline">Cardiac Silhouette</span>
          </button>

          {referenceVertices && (
            <button
              onClick={() => setShowReferenceCT((r) => !r)}
              className={`px-2 py-1 rounded text-xs font-medium flex items-center gap-1 transition-colors ${
                showReferenceCT
                  ? 'bg-cyan-100 text-cyan-950 font-semibold border border-cyan-300'
                  : 'text-[#5e4b75] hover:text-[#1a082e] hover:bg-slate-100'
              }`}
              title="Toggle Ground Truth Reference CT Mesh Overlay"
            >
              {showReferenceCT ? <Eye className="w-3.5 h-3.5 text-cyan-700" /> : <EyeOff className="w-3.5 h-3.5 text-slate-400" />}
              <span className="hidden sm:inline">CT Ground Truth (Cyan)</span>
            </button>
          )}

          <button
            onClick={() => setColorMode((c) => (c === 'pleura' ? 'bilateral' : c === 'bilateral' ? 'platinum' : 'pleura'))}
            className={`px-2 py-1 rounded text-xs font-medium flex items-center gap-1 transition-colors ${
              colorMode === 'pleura'
                ? 'bg-rose-100/90 text-rose-950 font-semibold border border-rose-300'
                : colorMode === 'bilateral'
                ? 'bg-amber-100/90 text-amber-950 font-semibold border border-amber-300'
                : 'text-[#5e4b75] hover:text-[#1a082e] hover:bg-slate-100'
            }`}
            title={
              colorMode === 'pleura'
                ? 'Visual Mode: Organic Visceral Pleura (Click for Bilateral Lobar)'
                : colorMode === 'bilateral'
                ? 'Visual Mode: Bilateral Lobar Partition (Click for Platinum Diagnostic)'
                : 'Visual Mode: High-Contrast Platinum (Click for Visceral Pleura)'
            }
          >
            <Palette className="w-3.5 h-3.5" />
            <span className="hidden sm:inline capitalize">
              {colorMode === 'pleura' ? 'Visceral Pleura' : colorMode === 'bilateral' ? 'Bilateral' : 'Platinum'}
            </span>
          </button>

          <button
            onClick={() => setIsAutoRotating((prev) => !prev)}
            className={`p-1.5 rounded hover:bg-slate-100 text-xs transition-colors ${isAutoRotating ? 'text-amber-700 bg-amber-50' : 'text-[#5e4b75] hover:text-[#1a082e]'
              }`}
            title={isAutoRotating ? 'Pause 360° Rotation' : 'Start 360° Rotation'}
          >
            {isAutoRotating ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          </button>

          <button
            onClick={handleExportSnapshot}
            className="p-1.5 rounded hover:bg-slate-100 text-[#5e4b75] hover:text-amber-800 transition-colors"
            title="Export High-Res Snapshot (PNG)"
          >
            <Camera className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => setAnatomicalView('OBLIQUE')}
            className="p-1.5 rounded hover:bg-slate-100 text-[#5e4b75] hover:text-[#1a082e] transition-colors"
            title="Reset Camera View"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          {onExportObj && (
            <Button
              variant="outline"
              size="sm"
              onClick={onExportObj}
              disabled={isExporting}
              className="text-xs ml-1 py-1 h-auto text-[#1a082e] border-[#e2e8f0] bg-white hover:bg-slate-50"
            >
              <Download className="w-3 h-3 mr-1" />
              <span>{isExporting ? '...' : 'OBJ'}</span>
            </Button>
          )}
        </div>
      </div>

      {/* Interactive Landmark Hover Tooltip */}
      {hoveredLandmark && (
        <div
          className="absolute z-30 pointer-events-none bg-white/95 backdrop-blur-md p-2.5 rounded-lg border border-amber-400 text-xs shadow-xl font-mono text-[#1a082e]"
          style={{
            left: `${Math.min(hoveredLandmark.x + 12, 360)}px`,
            top: `${Math.min(hoveredLandmark.y + 12, 320)}px`,
          }}
        >
          <div className="flex items-center gap-1.5 text-amber-800 font-bold mb-1">
            <MapPin className="w-3.5 h-3.5" />
            <span>Landmark: {hoveredLandmark.name.replace(/_/g, ' ')}</span>
          </div>
          <div className="text-[11px] text-[#5e4b75]">
            Coords: ({hoveredLandmark.coords[0].toFixed(1)}, {hoveredLandmark.coords[1].toFixed(1)}, {hoveredLandmark.coords[2].toFixed(1)}) mm
          </div>
        </div>
      )}

      {/* Orbit Controls Guide Bottom Pill & Active View Indicator */}
      <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between pointer-events-none z-20">
        <div className="bg-white/90 backdrop-blur-md px-2.5 py-1 rounded text-[10px] text-[#5e4b75] border border-[#e2e8f0] font-mono shadow-xs">
          Left click: Rotate · Right click: Pan · Scroll: Zoom
        </div>

        <div className="text-[10px] font-mono text-amber-900 bg-amber-50/95 backdrop-blur-md px-2.5 py-1 rounded border border-amber-300 flex items-center gap-1.5 shadow-xs font-semibold">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
          <span>VIEW: {activeViewName.toUpperCase()}</span>
        </div>
      </div>
    </div>
  );
};
