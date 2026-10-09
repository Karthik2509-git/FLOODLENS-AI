# Environment Audit (local Windows machine)

Reported by the developer's Antigravity environment inspection on 2026-10-10. This is a record of local command output supplied in the project conversation, not an independent remote verification.

| Component | Reported result |
|---|---|
| Project folder | `C:\Users\KARTHIK V\OneDrive\Desktop\FLOODLENS-AI` |
| Python default | 3.14.3 (`C:\Python314\python.exe`) |
| Python alternative | 3.10.11 (`...\Python310\python.exe`) |
| Node.js | v24.14.0 |
| npm | 11.9.0 |
| Git | 2.53.0.windows.2 |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM (as reported by `nvidia-smi`) |

## Reported checks

- Git repository initialized locally.
- Ignore patterns tested for environment files, `node_modules`, raster files, model weights, and Python cache.
- No large ML packages or model weights installed during the foundation audit.

## Notes

- Python 3.10 was proposed by the local agent as a compatibility option; package compatibility must be confirmed against the selected model and geospatial dependencies before setup.
- CUDA/PyTorch compatibility has not yet been tested.
- No inference, dataset download, backend, frontend, or cloud deployment has been verified at this stage.
