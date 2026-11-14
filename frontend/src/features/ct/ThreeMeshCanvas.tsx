import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { CTAnnotationItem } from '@/types/ct';
import { ctApi } from '@/services/api/ctApi';
import { RefreshCw, RotateCcw } from 'lucide-react';
import { Button } from '@/components/common/Button';

interface ThreeMeshCanvasProps {
  scanId: string;
  structure: string;
  annotations: CTAnnotationItem[];
  selectedNoduleId?: string | null;
  onSelectNodule?: (noduleId: string) => void;
}

export const ThreeMeshCanvas: React.FC<ThreeMeshCanvasProps> = ({
  scanId,
  structure,
  annotations,
  selectedNoduleId,
  onSelectNodule,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [vertexCount, setVertexCount] = useState(0);

  // References to preserve Three.js instances
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshGroupRef = useRef<THREE.Group | null>(null);
  const noduleGroupRef = useRef<THREE.Group | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  // 1. Initialize Scene, Camera, Renderer, Controls
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070b14);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 5000);
    camera.position.set(0, -450, 200); // Standard anterior-oblique view
    camera.up.set(0, 0, 1); // Z-axis pointing up (anatomical craniocaudal)

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = false;
    rendererRef.current = renderer;

    container.appendChild(renderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 2000;
    controls.minDistance = 50;
    controlsRef.current = controls;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.2);
    dirLight1.position.set(200, 300, 400);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x14b8a6, 0.8);
    dirLight2.position.set(-200, -300, -200);
    scene.add(dirLight2);

    // Groups
    const meshGroup = new THREE.Group();
    scene.add(meshGroup);
    meshGroupRef.current = meshGroup;

    const noduleGroup = new THREE.Group();
    scene.add(noduleGroup);
    noduleGroupRef.current = noduleGroup;

    // Animation Loop
    const animate = () => {
      animFrameIdRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize Handler
    const handleResize = () => {
      if (!container || !rendererRef.current) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };

    // Nodule click interaction
    const handleClick = (e: MouseEvent) => {
      if (!onSelectNodule || !container || !rendererRef.current || !noduleGroupRef.current) return;
      const rect = container.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(noduleGroupRef.current.children, true);
      if (intersects.length > 0 && intersects[0].object.userData?.noduleId) {
        onSelectNodule(intersects[0].object.userData.noduleId);
      }
    };
    container.addEventListener('click', handleClick);

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    // Cleanup
    return () => {
      container.removeEventListener('click', handleClick);
      resizeObserver.disconnect();
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    };
  }, []);

  // 2. Fetch and Load OBJ Mesh from Backend
  useEffect(() => {
    let isCancelled = false;
    setIsLoading(true);
    setError(null);

    const loadMesh = async () => {
      try {
        const objText = await ctApi.getMeshObj(scanId, structure, 2, true);
        if (isCancelled) return;

        const loader = new OBJLoader();
        const obj = loader.parse(objText);

        if (meshGroupRef.current) {
          // Dispose previous geometries and materials
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

          let totalVerts = 0;
          let meshColor = 0x06b6d4; // Cyan for bilateral lungs
          if (structure.includes('heart')) {
            meshColor = 0xf43f5e; // Crimson rose for heart
          } else if (structure.includes('airway') || structure.includes('trachea')) {
            meshColor = 0xf59e0b; // Amber for airways
          } else if (structure.includes('left_lung')) {
            meshColor = 0x38bdf8; // Sky blue for left lung
          } else if (structure.includes('right_lung')) {
            meshColor = 0x2dd4bf; // Teal for right lung
          }

          const material = new THREE.MeshStandardMaterial({
            color: meshColor,
            roughness: 0.35,
            metalness: 0.15,
            transparent: true,
            opacity: structure.includes('heart') ? 0.92 : 0.85,
            side: THREE.DoubleSide,
            wireframe: false,
          });

          obj.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
              const mesh = child as THREE.Mesh;
              mesh.material = material;
              mesh.geometry.computeVertexNormals();
              totalVerts += mesh.geometry.attributes.position.count;
            }
          });

          meshGroupRef.current.add(obj);
          setVertexCount(totalVerts);

          // Center bounding box to viewport
          const box = new THREE.Box3().setFromObject(obj);
          const center = box.getCenter(new THREE.Vector3());
          meshGroupRef.current.position.sub(center); // Recenter at origin
          if (noduleGroupRef.current) {
            noduleGroupRef.current.position.copy(meshGroupRef.current.position);
          }
        }
      } catch (err: any) {
        if (!isCancelled) {
          setError(err.message || 'Failed to stream 3D OBJ mesh from backend');
        }
      } finally {
        if (!isCancelled) {
          setIsLoading(false);
        }
      }
    };

    loadMesh();

    return () => {
      isCancelled = true;
    };
  }, [scanId, structure]);

  // 3. Render 3D Nodule Spheres from LUNA16 Annotations
  useEffect(() => {
    if (!noduleGroupRef.current || !meshGroupRef.current) return;

    // Clear previous nodule spheres
    while (noduleGroupRef.current.children.length > 0) {
      const child = noduleGroupRef.current.children[0] as THREE.Mesh;
      noduleGroupRef.current.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) (child.material as THREE.Material).dispose();
    }

    // Offset nodule group by the same centering offset as mesh group
    noduleGroupRef.current.position.copy(meshGroupRef.current.position);

    annotations.forEach((nodule) => {
      const isSelected = selectedNoduleId === nodule.nodule_id;
      const radius = Math.max((nodule.diameter_mm || 6.0) / 2.0, 3.0);

      const sphereGeom = new THREE.SphereGeometry(radius, 16, 16);
      const sphereMat = new THREE.MeshStandardMaterial({
        color: isSelected ? 0xef4444 : 0xf59e0b,
        emissive: isSelected ? 0x991b1b : 0x78350f,
        roughness: 0.2,
        metalness: 0.5,
      });

      const sphere = new THREE.Mesh(sphereGeom, sphereMat);
      // Nodule coordinates are in physical millimeters (X, Y, Z)
      sphere.position.set(nodule.coordX_mm, nodule.coordY_mm, nodule.coordZ_mm);
      sphere.userData = { noduleId: nodule.nodule_id };

      noduleGroupRef.current?.add(sphere);
    });
  }, [annotations, selectedNoduleId]);

  const resetCamera = () => {
    if (!controlsRef.current) return;
    controlsRef.current.reset();
  };

  return (
    <div className="relative w-full h-full min-h-[420px] rounded-xl overflow-hidden border border-medical-border bg-medical-bg">
      <div ref={mountRef} className="w-full h-full three-canvas-container" />

      {/* Top Floating Controls */}
      <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 text-xs font-mono text-slate-300">
        <span className="w-2 h-2 rounded-full bg-medical-cyan animate-pulse" />
        <span>Three.js Engine</span>
        <span className="text-medical-subtle">|</span>
        <span>{vertexCount.toLocaleString()} Vertices</span>
      </div>

      <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-black/60 backdrop-blur-md p-1 rounded-lg border border-white/10">
        <Button
          variant="ghost"
          size="sm"
          className="text-white hover:text-medical-cyan p-1.5 h-auto"
          onClick={resetCamera}
          title="Reset Camera View"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </Button>
      </div>

      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-medical-bg/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3">
          <RefreshCw className="w-8 h-8 text-medical-cyan animate-spin" />
          <p className="text-xs font-mono text-medical-cyan">
            Streaming Watertight Marching Cubes Mesh...
          </p>
        </div>
      )}

      {/* Error Overlay */}
      {error && (
        <div className="absolute inset-0 bg-medical-bg/90 p-6 flex flex-col items-center justify-center text-center">
          <p className="text-sm font-semibold text-rose-400 mb-1">Mesh Stream Error</p>
          <p className="text-xs text-medical-muted max-w-md">{error}</p>
        </div>
      )}

      {/* Bottom Hint */}
      <div className="absolute bottom-3 left-3 text-[10px] text-medical-subtle font-mono pointer-events-none bg-black/40 px-2 py-0.5 rounded">
        Left Click: Rotate · Right Click: Pan · Scroll: Zoom
      </div>
    </div>
  );
};
