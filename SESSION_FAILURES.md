## Session Failures — 2026-11-12

- **pytest command**: Running `pytest.exe` directly was blocked by Windows Application Control policy (`ApplicationFailedException`).
  **Adaptation**: Use `python -m pytest backend/tests` directly through Python interpreter.

- **torch.load default in PyTorch 2.6+**: `weights_only=True` default caused `_pickle.UnpicklingError` due to numpy scalar metadata in checkpoint dictionaries.
  **Adaptation**: Pass `weights_only=False` explicitly when loading trusted historical checkpoints.

- **FastAPI Form parameters in multipart requests**: Sending `model_id` in query parameters for `/xray/predict-binary` resulted in FastAPI applying the default Form parameter.
  **Adaptation**: Form parameters in multipart endpoints must be encoded as explicit form body parts rather than URL query parameters.

