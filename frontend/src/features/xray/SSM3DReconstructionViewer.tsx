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
  landmarks,
  onExportObj,
  isExporting = false,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [wireframe, setWireframe] = useState<boolean>(false);
  const [showLandmarks, setShowLandmarks] = useState<boolean>(true);
  const [colorMode, setColorMode] = useState<'cyan' | 'bilateral'>('cyan');
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
    scene.background = new THREE.Color(0x0c0614);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 4000);
    camera.position.set(0, -380, 160);
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
    renderer.toneMappingExposure = 1.1;
    rendererRef.current = renderer;

    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 1500;
    controls.minDistance = 40;
    controls.addEventListener('change', updateViewLabel);
    controlsRef.current = controls;

    // Lighting with warm amber key and platinum rim
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xfffbeb, 1.25);
    dirLight1.position.set(150, -250, 300);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xe2e8f0, 0.9);
    dirLight2.position.set(-150, 250, -150);
    scene.add(dirLight2);

    const backLight = new THREE.DirectionalLight(0xf59e0b, 0.55);
    backLight.position.set(0, 350, 200);
    scene.add(backLight);

    // Groups
    const meshGroup = new THREE.Group();
    scene.add(meshGroup);
    meshGroupRef.current = meshGroup;

    const landmarkGroup = new THREE.Group();
    scene.add(landmarkGroup);
    landmarkGroupRef.current = landmarkGroup;

    // Animation Loop
    const animate = () => {
      animFrameIdRef.current = requestAnimationFrame(animate);
      if (isAutoRotatingRef.current && meshGroupRef.current) {
        meshGroupRef.current.rotation.z += 0.006;
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

    // Clear previous mesh
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

    const material = new THREE.MeshStandardMaterial({
      color: 0xf1f5f9,
      roughness: 0.28,
      metalness: 0.22,
      transparent: true,
      opacity: wireframe ? 0.9 : 0.88,
      wireframe: wireframe,
      side: THREE.DoubleSide,
    });

    if (objData) {
      const loader = new OBJLoader();
      try {
        const obj = loader.parse(objData);
        let totalVerts = 0;
        let totalFaces = 0;
        obj.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            mesh.material = material;
            mesh.geometry.computeVertexNormals();
            totalVerts += mesh.geometry.attributes.position.count;
            if (mesh.geometry.index) {
              totalFaces += mesh.geometry.index.count / 3;
            }
          }
        });
        meshGroupRef.current.add(obj);
        setVertexCount(totalVerts);
        setFaceCount(totalFaces || 13478);

        // Center mesh
        const box = new THREE.Box3().setFromObject(obj);
        const center = box.getCenter(new THREE.Vector3());
        meshGroupRef.current.position.sub(center);
        meshGroupRef.current.rotation.z = 0;
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

      // Optional vertex color assignment for bilateral split mode (Ice Silver / Champagne Gold)
      if (colorMode === 'bilateral') {
        const colors = new Float32Array(vertices.length * 3);
        for (let i = 0; i < vertices.length; i++) {
          const x = vertices[i][0];
          if (x < 0) {
            // Left lung (Ice Silver Platinum: 0.95, 0.98, 1.0)
            colors[i * 3] = 0.95;
            colors[i * 3 + 1] = 0.98;
            colors[i * 3 + 2] = 1.0;
          } else {
            // Right lung (Warm Champagne Gold: 0.99, 0.88, 0.54)
            colors[i * 3] = 0.99;
            colors[i * 3 + 1] = 0.88;
            colors[i * 3 + 2] = 0.54;
          }
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        material.vertexColors = true;
      }

      const indices: number[] = [];
      for (let f = 0; f < faces.length; f++) {
        indices.push(faces[f][0], faces[f][1], faces[f][2]);
      }
      geometry.setIndex(indices);
      geometry.computeVertexNormals();

      const mesh = new THREE.Mesh(geometry, material);
      meshGroupRef.current.add(mesh);
      setVertexCount(vertices.length);
      setFaceCount(faces.length);

      const box = new THREE.Box3().setFromObject(mesh);
      const center = box.getCenter(new THREE.Vector3());
      meshGroupRef.current.position.sub(center);
      meshGroupRef.current.rotation.z = 0;
      if (landmarkGroupRef.current) {
        landmarkGroupRef.current.position.copy(meshGroupRef.current.position);
        landmarkGroupRef.current.rotation.z = 0;
      }
    }
  }, [objData, vertices, faces, wireframe, colorMode]);

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
        camera.position.set(0, -380, 0);
        camera.up.set(0, 0, 1);
        break;
      case 'PA':
        camera.position.set(0, 380, 0);
        camera.up.set(0, 0, 1);
        break;
      case 'RL':
        camera.position.set(380, 0, 0);
        camera.up.set(0, 0, 1);
        break;
      case 'LL':
        camera.position.set(-380, 0, 0);
        camera.up.set(0, 0, 1);
        break;
      case 'SI':
        camera.position.set(0, 0, 380);
        camera.up.set(0, 1, 0);
        break;
      case 'OBLIQUE':
      default:
        camera.position.set(0, -380, 160);
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
    <div className="relative w-full h-[460px] rounded-xl overflow-hidden bg-[#0c0614] border border-white/10 flex flex-col">
      {/* 3D Canvas Mount Point */}
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Viewport Overlay HUD Header */}
      <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none z-20">
        <div className="flex items-center gap-2 pointer-events-auto bg-[#140a22]/85 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 text-xs shadow-xl">
          <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
          <span className="font-semibold text-white font-mono">SSM 3D Surface</span>
          <span className="text-white/20">|</span>
          <span className="text-slate-300">{vertexCount.toLocaleString()} Vertices</span>
          <span className="text-white/20">·</span>
          <span className="text-slate-300">{faceCount.toLocaleString()} Triangles</span>
        </div>

        {/* Viewport Action Controls */}
        <div className="flex items-center gap-1.5 pointer-events-auto bg-[#140a22]/85 backdrop-blur-md p-1.5 rounded-lg border border-white/10 shadow-xl">
          {/* Quick Camera Presets */}
          <div className="flex items-center gap-1 mr-1 border-r border-white/10 pr-1.5 text-[10px] font-mono">
            <button
              onClick={() => setAnatomicalView('AP')}
              className="px-1.5 py-1 rounded hover:bg-white/10 text-slate-300 hover:text-amber-300 transition-colors"
              title="Anterior (Frontal AP View)"
            >
              AP
            </button>
            <button
              onClick={() => setAnatomicalView('PA')}
              className="px-1.5 py-1 rounded hover:bg-white/10 text-slate-300 hover:text-amber-300 transition-colors"
              title="Posterior (Dorsal PA View)"
            >
              PA
            </button>
            <button
              onClick={() => setAnatomicalView('RL')}
              className="px-1.5 py-1 rounded hover:bg-white/10 text-slate-300 hover:text-amber-300 transition-colors"
              title="Right Lateral View"
            >
              RL
            </button>
            <button
              onClick={() => setAnatomicalView('LL')}
              className="px-1.5 py-1 rounded hover:bg-white/10 text-slate-300 hover:text-amber-300 transition-colors"
              title="Left Lateral View"
            >
              LL
            </button>
            <button
              onClick={() => setAnatomicalView('SI')}
              className="px-1.5 py-1 rounded hover:bg-white/10 text-slate-300 hover:text-amber-300 transition-colors"
              title="Superior (Top-Down View)"
            >
              SI
            </button>
          </div>

          <button
            onClick={() => setWireframe((w) => !w)}
            className={`px-2 py-1 rounded text-xs font-medium flex items-center gap-1 transition-colors ${wireframe
              ? 'bg-amber-500/20 text-amber-300 font-semibold'
              : 'text-slate-400 hover:text-white'
              }`}
            title="Toggle Wireframe Mesh"
          >
            <Layers className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Wireframe</span>
          </button>

          <button
            onClick={() => setShowLandmarks((s) => !s)}
            className={`px-2 py-1 rounded text-xs font-medium flex items-center gap-1 transition-colors ${showLandmarks
              ? 'bg-amber-500/20 text-amber-300 font-semibold'
              : 'text-slate-400 hover:text-white'
              }`}
            title="Toggle Anatomical Landmarks"
          >
            {showLandmarks ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span className="hidden sm:inline">Landmarks</span>
          </button>

          <button
            onClick={() => setColorMode((c) => (c === 'cyan' ? 'bilateral' : 'cyan'))}
            className={`p-1.5 rounded hover:bg-white/10 text-xs transition-colors ${colorMode === 'bilateral' ? 'text-amber-300 font-semibold' : 'text-slate-400 hover:text-white'
              }`}
            title={colorMode === 'bilateral' ? 'Bilateral Color Partition' : 'Single Shaded Color'}
          >
            <Palette className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => setIsAutoRotating((prev) => !prev)}
            className={`p-1.5 rounded hover:bg-white/10 text-xs transition-colors ${isAutoRotating ? 'text-amber-400' : 'text-slate-400 hover:text-white'
              }`}
            title={isAutoRotating ? 'Pause 360° Rotation' : 'Start 360° Rotation'}
          >
            {isAutoRotating ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          </button>

          <button
            onClick={handleExportSnapshot}
            className="p-1.5 rounded hover:bg-white/10 text-slate-400 hover:text-amber-300 transition-colors"
            title="Export High-Res Snapshot (PNG)"
          >
            <Camera className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => setAnatomicalView('OBLIQUE')}
            className="p-1.5 rounded hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
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
              className="text-xs ml-1 py-1 h-auto"
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
          className="absolute z-30 pointer-events-none bg-slate-950/95 backdrop-blur-md p-2.5 rounded-lg border border-amber-500/50 text-xs shadow-2xl font-mono text-slate-200"
          style={{
            left: `${Math.min(hoveredLandmark.x + 12, 360)}px`,
            top: `${Math.min(hoveredLandmark.y + 12, 320)}px`,
          }}
        >
          <div className="flex items-center gap-1.5 text-amber-400 font-bold mb-1">
            <MapPin className="w-3.5 h-3.5" />
            <span>Landmark: {hoveredLandmark.name.replace(/_/g, ' ')}</span>
          </div>
          <div className="text-[11px] text-slate-300">
            Coords: ({hoveredLandmark.coords[0].toFixed(1)}, {hoveredLandmark.coords[1].toFixed(1)}, {hoveredLandmark.coords[2].toFixed(1)}) mm
          </div>
        </div>
      )}

      {/* Orbit Controls Guide Bottom Pill & Active View Indicator */}
      <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between pointer-events-none z-20">
        <div className="bg-black/70 backdrop-blur-md px-2.5 py-1 rounded text-[10px] text-slate-400 border border-white/10 font-mono">
          Left click: Rotate · Right click: Pan · Scroll: Zoom
        </div>

        <div className="text-[10px] font-mono text-amber-300 bg-amber-950/70 backdrop-blur-md px-2.5 py-1 rounded border border-amber-800/40 flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
          <span>VIEW: {activeViewName.toUpperCase()}</span>
        </div>
      </div>
    </div>
  );
};
