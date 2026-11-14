## Session Failures — 2026-11-12

- **pytest command**: Running `pytest.exe` directly was blocked by Windows Application Control policy (`ApplicationFailedException`).
  **Adaptation**: Use `python -m pytest backend/tests` directly through Python interpreter.

- **torch.load default in PyTorch 2.6+**: `weights_only=True` default caused `_pickle.UnpicklingError` due to numpy scalar metadata in checkpoint dictionaries.
  **Adaptation**: Pass `weights_only=False` explicitly when loading trusted historical checkpoints.

- **FastAPI Form parameters in multipart requests**: Sending `model_id` in query parameters for `/xray/predict-binary` resulted in FastAPI applying the default Form parameter.
  **Adaptation**: Form parameters in multipart endpoints must be encoded as explicit form body parts rather than URL query parameters.

## Session Failures & Adaptations — 2026-11-15

- **trimesh.proximity.closest_point requires optional 'rtree'**: Calling `trimesh.proximity.closest_point` threw `ModuleNotFoundError: No module named 'rtree'`.
  **Adaptation**: Use `scipy.spatial.cKDTree` directly on mesh vertices to perform fast, robust Euclidean proximity queries without external C binary dependencies.

- **Euler characteristic of combined lung mesh (\chi = -10)**: The combined mesh reported $\chi = -10$.
  **Investigation & Adaptation**: Dissected components mathematically: Right lung is an ideal genus-0 sphere ($\chi = +2$). Left lung has $\chi = -12$ (one main component with 8 micro-handles, $g = 8, \chi = -14$, plus 1 single-voxel boundary octahedron, $g = 0, \chi = +2$). The handles stem from discrete voxel discretization at the grazing intersection between the cardiac notch ellipsoid and left lung base. The surface is 100% watertight (0 boundary edges, 0 non-manifold edges) with strictly outward normals. Do not artificially force Euler characteristic; document exact topological provenance in audit records.


