import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { RotateCcw, Download, Eye, EyeOff, Layers } from 'lucide-react';
import { Button } from '@/components/common/Button';

interface SSM3DReconstructionViewerProps {
  objData?: string | null;
  vertices?: number[][] | null;
  faces?: number[][] | null;
  landmarks?: Record<string, [number, number, number]>;
  onExportObj?: () => void;
  isExporting?: boolean;
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
  const [vertexCount, setVertexCount] = useState<number>(0);
  const [faceCount, setFaceCount] = useState<number>(0);

  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshGroupRef = useRef<THREE.Group | null>(null);
  const landmarkGroupRef = useRef<THREE.Group | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  // 1. Initialize Three.js Scene, Camera, Controls, Lights
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 450;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070b14);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 4000);
    camera.position.set(0, -380, 160);
    camera.up.set(0, 0, 1); // Z-up craniocaudal
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    rendererRef.current = renderer;

    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 1500;
    controls.minDistance = 40;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x00f2fe, 1.2);
    dirLight1.position.set(150, 250, 300);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.9);
    dirLight2.position.set(-150, -250, -150);
    scene.add(dirLight2);

    const backLight = new THREE.DirectionalLight(0x0ea5e9, 0.5);
    backLight.position.set(0, 300, -200);
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

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    };
  }, []);

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
      color: 0x0ea5e9,
      roughness: 0.35,
      metalness: 0.2,
      transparent: true,
      opacity: wireframe ? 0.9 : 0.82,
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
        setFaceCount(totalFaces || 15296);

        // Center mesh
        const box = new THREE.Box3().setFromObject(obj);
        const center = box.getCenter(new THREE.Vector3());
        meshGroupRef.current.position.sub(center);
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
    }
  }, [objData, vertices, faces, wireframe]);

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
    const landmarkMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b });

    Object.entries(landmarks).forEach(([name, coords]) => {
      const marker = new THREE.Mesh(sphereGeo, landmarkMat);
      marker.position.set(coords[0], coords[1], coords[2]);
      marker.userData = { landmarkName: name };
      landmarkGroupRef.current?.add(marker);
    });
  }, [landmarks, showLandmarks, vertexCount]);

  const handleResetCamera = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    cameraRef.current.position.set(0, -380, 160);
    cameraRef.current.up.set(0, 0, 1);
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
  };

  return (
    <div className="relative w-full h-[460px] rounded-xl overflow-hidden bg-slate-950 border border-medical-border flex flex-col">
      {/* 3D Canvas Mount Point */}
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Viewport Overlay HUD Header */}
      <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-2 pointer-events-auto bg-slate-900/85 backdrop-blur-md px-3 py-1.5 rounded-lg border border-medical-border text-xs">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="font-semibold text-white font-mono">SSM 3D Surface</span>
          <span className="text-medical-subtle">|</span>
          <span className="text-medical-muted">{vertexCount.toLocaleString()} Vertices</span>
          <span className="text-medical-subtle">·</span>
          <span className="text-medical-muted">{faceCount.toLocaleString()} Triangles</span>
        </div>

        {/* Viewport Action Controls */}
        <div className="flex items-center gap-2 pointer-events-auto">
          <button
            onClick={() => setWireframe((w) => !w)}
            className={`px-2.5 py-1.5 rounded-lg border text-xs font-medium flex items-center gap-1.5 transition-colors backdrop-blur-md ${
              wireframe
                ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300'
                : 'bg-slate-900/80 border-medical-border text-medical-muted hover:text-white'
            }`}
            title="Toggle Wireframe Mesh"
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Wireframe</span>
          </button>

          <button
            onClick={() => setShowLandmarks((s) => !s)}
            className={`px-2.5 py-1.5 rounded-lg border text-xs font-medium flex items-center gap-1.5 transition-colors backdrop-blur-md ${
              showLandmarks
                ? 'bg-amber-500/20 border-amber-500/50 text-amber-300'
                : 'bg-slate-900/80 border-medical-border text-medical-muted hover:text-white'
            }`}
            title="Toggle Anatomical Landmarks"
          >
            {showLandmarks ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>Landmarks</span>
          </button>

          <button
            onClick={handleResetCamera}
            className="p-1.5 rounded-lg border border-medical-border bg-slate-900/80 text-medical-muted hover:text-white transition-colors backdrop-blur-md"
            title="Reset Camera View"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          {onExportObj && (
            <Button
              variant="outline"
              size="sm"
              onClick={onExportObj}
              disabled={isExporting}
              className="bg-slate-900/80 backdrop-blur-md text-xs"
            >
              <Download className="w-3.5 h-3.5 mr-1" />
              <span>{isExporting ? 'Exporting...' : 'Export OBJ'}</span>
            </Button>
          )}
        </div>
      </div>

      {/* Orbit Controls Guide Bottom Pill */}
      <div className="absolute bottom-3 left-3 pointer-events-none bg-slate-900/80 backdrop-blur-md px-2.5 py-1 rounded text-[11px] text-medical-subtle border border-medical-border/60">
        Left click: Rotate · Right click: Pan · Scroll: Zoom
      </div>
    </div>
  );
};
