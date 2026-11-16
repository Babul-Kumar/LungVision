import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { CTAnnotationItem } from '@/types/ct';
import { ctApi } from '@/services/api/ctApi';
import { RefreshCw, RotateCcw, Camera, Play, Pause, Crosshair } from 'lucide-react';
import { Button } from '@/components/common/Button';

interface ThreeMeshCanvasProps {
  scanId: string;
  structure: string;
  annotations: CTAnnotationItem[];
  selectedNoduleId?: string | null;
  onSelectNodule?: (noduleId: string) => void;
}

interface HoveredNoduleInfo {
  nodule: CTAnnotationItem;
  x: number;
  y: number;
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
  const [isAutoRotating, setIsAutoRotating] = useState(false);
  const [activeViewName, setActiveViewName] = useState<string>('Oblique 3D');
  const [hoveredNodule, setHoveredNodule] = useState<HoveredNoduleInfo | null>(null);

  // References to preserve Three.js instances
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const rootGroupRef = useRef<THREE.Group | null>(null);
  const meshGroupRef = useRef<THREE.Group | null>(null);
  const noduleGroupRef = useRef<THREE.Group | null>(null);
  const animFrameIdRef = useRef<number | null>(null);
  const isAutoRotatingRef = useRef(false);

  // Sync ref with state for animation loop
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

  // 1. Initialize Scene, Camera, Renderer, Controls
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x160a28);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 5000);
    camera.position.set(0, -450, 200); // Standard anterior-oblique view
    camera.up.set(0, 0, 1); // Z-axis pointing up (anatomical craniocaudal)
    cameraRef.current = camera;

    // Renderer with high-DPI and preserveDrawingBuffer for snapshots
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

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 2000;
    controls.minDistance = 50;
    controls.addEventListener('change', updateViewLabel);
    controlsRef.current = controls;

    // Lights: balanced 3-point clinical lighting with warm amber key & platinum rim
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xfffbeb, 1.25);
    dirLight1.position.set(200, -350, 400);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xe2e8f0, 0.9);
    dirLight2.position.set(-200, 350, -200);
    scene.add(dirLight2);

    const rimLight = new THREE.DirectionalLight(0xf59e0b, 0.55);
    rimLight.position.set(0, 400, 250);
    scene.add(rimLight);

    // Root group for anatomical registration & shared rotation
    const rootGroup = new THREE.Group();
    scene.add(rootGroup);
    rootGroupRef.current = rootGroup;

    // Groups parented under rootGroup
    const meshGroup = new THREE.Group();
    rootGroup.add(meshGroup);
    meshGroupRef.current = meshGroup;

    const noduleGroup = new THREE.Group();
    rootGroup.add(noduleGroup);
    noduleGroupRef.current = noduleGroup;

    // Animation Loop
    const animate = () => {
      animFrameIdRef.current = requestAnimationFrame(animate);
      if (isAutoRotatingRef.current && rootGroupRef.current) {
        rootGroupRef.current.rotation.z += 0.006;
      }
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize Handler
    const handleResize = () => {
      if (!container || !rendererRef.current || !cameraRef.current) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };

    // Raycasting for click & hover
    const getIntersectedNodule = (e: MouseEvent): CTAnnotationItem | null => {
      if (!container || !rendererRef.current || !noduleGroupRef.current || !cameraRef.current) {
        return null;
      }
      const rect = container.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, cameraRef.current);
      const intersects = raycaster.intersectObjects(noduleGroupRef.current.children, true);
      if (intersects.length > 0 && intersects[0].object.userData?.noduleId) {
        const id = intersects[0].object.userData.noduleId;
        return annotations.find((a) => a.nodule_id === id) || null;
      }
      return null;
    };

    const handleClick = (e: MouseEvent) => {
      const nodule = getIntersectedNodule(e);
      if (nodule && onSelectNodule) {
        onSelectNodule(nodule.nodule_id);
      }
    };

    let mouseMoveRaf: number | null = null;
    const handleMouseMove = (e: MouseEvent) => {
      if (mouseMoveRaf) return;
      mouseMoveRaf = requestAnimationFrame(() => {
        mouseMoveRaf = null;
        const nodule = getIntersectedNodule(e);
        if (nodule && container) {
          const rect = container.getBoundingClientRect();
          setHoveredNodule({
            nodule,
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
          });
        } else {
          setHoveredNodule((prev) => (prev !== null ? null : prev));
        }
      });
    };

    container.addEventListener('click', handleClick);
    container.addEventListener('mousemove', handleMouseMove);

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    // Cleanup
    return () => {
      container.removeEventListener('click', handleClick);
      container.removeEventListener('mousemove', handleMouseMove);
      resizeObserver.disconnect();
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
      controls.removeEventListener('change', updateViewLabel);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    };
  }, [annotations, onSelectNodule, updateViewLabel]);

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
          let meshColor = 0xf4aba6; // Organic visceral pleura living pulmonary pink
          let opacity = 0.94;
          let clearcoatVal = 0.65;
          let roughnessVal = 0.32;
          let metalnessVal = 0.04;
          let transmissionVal = 0.16;

          if (structure.includes('heart')) {
            meshColor = 0xbe123c; // Crimson burgundy rose for heart
            opacity = 0.96;
            clearcoatVal = 0.6;
            roughnessVal = 0.25;
            transmissionVal = 0.05;
          } else if (structure.includes('airway') || structure.includes('trachea')) {
            meshColor = 0xf59e0b; // Rich Marigold amber for airways
            opacity = 0.95;
            clearcoatVal = 0.5;
            roughnessVal = 0.35;
            transmissionVal = 0.08;
          } else if (structure.includes('left_lung')) {
            meshColor = 0xf1f5f9; // Ice silver platinum left lung
            opacity = 0.94;
          } else if (structure.includes('right_lung')) {
            meshColor = 0xfef3c7; // Warm champagne gold right lung
            opacity = 0.94;
          }

          const material = new THREE.MeshPhysicalMaterial({
            color: meshColor,
            roughness: roughnessVal,
            metalness: metalnessVal,
            clearcoat: clearcoatVal,
            clearcoatRoughness: 0.12,
            transmission: transmissionVal,
            ior: 1.34,
            reflectivity: 0.5,
            sheen: 0.4,
            sheenColor: new THREE.Color(0xffccd5),
            transparent: true,
            opacity: opacity,
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

          // Reset rootGroup transform before calculating bounding box
          if (rootGroupRef.current) {
            rootGroupRef.current.position.set(0, 0, 0);
            rootGroupRef.current.rotation.set(0, 0, 0);
          }

          meshGroupRef.current.add(obj);
          setVertexCount(totalVerts);

          // Center bounding box to viewport via rootGroup offset
          const box = new THREE.Box3().setFromObject(obj);
          const center = box.getCenter(new THREE.Vector3());
          if (rootGroupRef.current) {
            rootGroupRef.current.position.set(-center.x, -center.y, -center.z);
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
    if (!noduleGroupRef.current) return;

    // Clear previous nodule spheres
    while (noduleGroupRef.current.children.length > 0) {
      const child = noduleGroupRef.current.children[0] as THREE.Mesh;
      noduleGroupRef.current.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) (child.material as THREE.Material).dispose();
    }

    annotations.forEach((nodule) => {
      const isSelected = selectedNoduleId === nodule.nodule_id;
      const radius = Math.max((nodule.diameter_mm || 6.0) / 2.0, 3.2);

      const sphereGeom = new THREE.SphereGeometry(radius, 20, 20);
      const sphereMat = new THREE.MeshStandardMaterial({
        color: isSelected ? 0xef4444 : 0xf59e0b,
        emissive: isSelected ? 0xdc2626 : 0xb45309,
        emissiveIntensity: isSelected ? 0.8 : 0.4,
        roughness: 0.2,
        metalness: 0.6,
      });

      const sphere = new THREE.Mesh(sphereGeom, sphereMat);
      // Nodule coordinates are in physical millimeters (X, Y, Z) matching the mesh space
      sphere.position.set(nodule.coordX_mm, nodule.coordY_mm, nodule.coordZ_mm);
      sphere.userData = { noduleId: nodule.nodule_id };

      noduleGroupRef.current?.add(sphere);
    });
  }, [annotations, selectedNoduleId]);

  // View angle preset dispatcher
  const setAnatomicalView = (view: 'AP' | 'PA' | 'RL' | 'LL' | 'SI' | 'OBLIQUE') => {
    if (!cameraRef.current || !controlsRef.current) return;
    const camera = cameraRef.current;
    const controls = controlsRef.current;

    controls.target.set(0, 0, 0);

    switch (view) {
      case 'AP': // Anterior-Posterior (Frontal)
        camera.position.set(0, -450, 0);
        camera.up.set(0, 0, 1);
        break;
      case 'PA': // Posterior-Anterior (Dorsal)
        camera.position.set(0, 450, 0);
        camera.up.set(0, 0, 1);
        break;
      case 'RL': // Right Lateral
        camera.position.set(450, 0, 0);
        camera.up.set(0, 0, 1);
        break;
      case 'LL': // Left Lateral
        camera.position.set(-450, 0, 0);
        camera.up.set(0, 0, 1);
        break;
      case 'SI': // Superior (Axial Cranial)
        camera.position.set(0, 0, 450);
        camera.up.set(0, 1, 0);
        break;
      case 'OBLIQUE': // 3/4 Perspective
      default:
        camera.position.set(0, -450, 200);
        camera.up.set(0, 0, 1);
        break;
    }

    controls.update();
    updateViewLabel();
  };

  // Export full-res snapshot PNG of the current 3D viewport
  const handleExportSnapshot = () => {
    if (!rendererRef.current) return;
    const dataUrl = rendererRef.current.domElement.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = `lungvision_3d_${structure}_${scanId}.png`;
    link.href = dataUrl;
    link.click();
  };

  return (
    <div className="relative w-full h-full min-h-[440px] rounded-xl overflow-hidden border border-[#e2e8f0] bg-[#160a28] shadow-inner">
      <div ref={mountRef} className="w-full h-full three-canvas-container" />

      {/* Top Left Floating Status Card */}
      <div className="absolute top-3 left-3 flex items-center gap-2 bg-white/95 backdrop-blur-md px-3 py-1.5 rounded-lg border border-[#e2e8f0] text-xs font-mono text-[#1a082e] shadow-md">
        <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
        <span className="font-semibold text-[#1a082e]">Three.js Engine</span>
        <span className="text-slate-300">|</span>
        <span className="text-[#5e4b75]">{vertexCount.toLocaleString()} Vertices</span>
      </div>

      {/* Top Right Floating Toolbar: Camera Presets & Actions */}
      <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-white/95 backdrop-blur-md p-1.5 rounded-lg border border-[#e2e8f0] shadow-md z-20">
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

        {/* Auto-Turntable Toggle */}
        <Button
          variant="ghost"
          size="sm"
          className={`p-1.5 h-auto transition-colors ${
            isAutoRotating ? 'text-amber-800 bg-amber-50 font-semibold' : 'text-[#5e4b75] hover:text-[#1a082e]'
          }`}
          onClick={() => setIsAutoRotating((prev) => !prev)}
          title={isAutoRotating ? 'Pause 360° Rotation' : 'Start 360° Rotation'}
        >
          {isAutoRotating ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
        </Button>

        {/* Export Snapshot */}
        <Button
          variant="ghost"
          size="sm"
          className="text-[#5e4b75] hover:text-amber-800 p-1.5 h-auto transition-colors"
          onClick={handleExportSnapshot}
          title="Export High-Res Snapshot (PNG)"
        >
          <Camera className="w-3.5 h-3.5" />
        </Button>

        {/* Reset Camera to Oblique 3/4 */}
        <Button
          variant="ghost"
          size="sm"
          className="text-[#5e4b75] hover:text-[#1a082e] p-1.5 h-auto transition-colors"
          onClick={() => setAnatomicalView('OBLIQUE')}
          title="Reset Camera View"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </Button>
      </div>

      {/* Interactive Nodule Hover Tooltip */}
      {hoveredNodule && (
        <div
          className="absolute z-30 pointer-events-none bg-white/95 backdrop-blur-md p-2.5 rounded-lg border border-amber-400 text-xs shadow-xl font-mono text-[#1a082e]"
          style={{
            left: `${Math.min(hoveredNodule.x + 12, 340)}px`,
            top: `${Math.min(hoveredNodule.y + 12, 300)}px`,
          }}
        >
          <div className="flex items-center gap-1.5 text-amber-800 font-bold mb-1">
            <Crosshair className="w-3.5 h-3.5" />
            <span>Nodule: {hoveredNodule.nodule.nodule_id}</span>
          </div>
          <div className="text-[11px] text-[#5e4b75] space-y-0.5">
            <div>Coords: ({hoveredNodule.nodule.coordX_mm.toFixed(1)}, {hoveredNodule.nodule.coordY_mm.toFixed(1)}, {hoveredNodule.nodule.coordZ_mm.toFixed(1)}) mm</div>
            <div>Diameter: <span className="text-[#1a082e] font-semibold">{hoveredNodule.nodule.diameter_mm.toFixed(1)} mm</span></div>
            <div className="text-[10px] text-emerald-700 font-semibold">Status: Registered in Lung Tissue</div>
          </div>
        </div>
      )}

      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-white/90 backdrop-blur-sm flex flex-col items-center justify-center gap-3 z-30">
          <RefreshCw className="w-8 h-8 text-amber-600 animate-spin" />
          <p className="text-xs font-mono text-amber-900 font-semibold">
            Streaming Watertight Marching Cubes Mesh...
          </p>
        </div>
      )}

      {/* Error Overlay */}
      {error && (
        <div className="absolute inset-0 bg-white/95 p-6 flex flex-col items-center justify-center text-center z-30">
          <p className="text-sm font-semibold text-rose-700 mb-1">Mesh Stream Error</p>
          <p className="text-xs text-[#5e4b75] max-w-md">{error}</p>
        </div>
      )}

      {/* Bottom HUD Bar: Active View Pill & Mouse Controls */}
      <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between pointer-events-none z-20">
        <div className="text-[10px] text-[#5e4b75] font-mono bg-white/90 backdrop-blur-sm px-2.5 py-1 rounded-md border border-[#e2e8f0] shadow-xs">
          Left Click: Rotate · Right Click: Pan · Scroll: Zoom
        </div>

        <div className="text-[10px] font-mono text-amber-900 bg-amber-50/95 backdrop-blur-sm px-2.5 py-1 rounded-md border border-amber-300 flex items-center gap-1.5 shadow-xs font-semibold">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
          <span>VIEW: {activeViewName.toUpperCase()}</span>
        </div>
      </div>
    </div>
  );
};
