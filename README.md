# FLOODLENS-AI

**Satellite-to-response flood intelligence** — a hackathon project for Multimodal AI Hackathon 2026, Track B: Mapping Flood Damage from Space.

## Mission

Turn georeferenced satellite imagery into a map-based flood situation overview that distinguishes observed image evidence from inferred consequences. The project aims to support situational awareness by mapping flood extent, contextualizing nearby infrastructure, and analyzing potentially disrupted road connectivity.

> **Research/demo tool only.** Outputs are not authoritative emergency guidance. Flood detections, infrastructure status, connectivity implications, and priorities must be verified by qualified responders. The system must communicate uncertainty and must not claim people are trapped or networks are down without supporting data.

## Planned workflow

1. Load compatible, georeferenced satellite imagery and preserve its spatial metadata.
2. Run a documented, pretrained flood-segmentation baseline on the correct input bands and preprocessing.
3. Evaluate available predictions against independent reference labels where possible.
4. Display flood masks and evidence on an interactive map.
5. Combine flood evidence with road/settlement context to estimate potential access disruption, explicitly labeling assumptions.
6. Generate an incident summary grounded in computed evidence, with sources and uncertainty.

## Proposed stack

- **ML/data:** Python, PyTorch, Rasterio, NumPy
- **API:** FastAPI
- **Frontend:** React, TypeScript, Tailwind CSS, Leaflet
- **Network analysis:** NetworkX (planned)

Technology choices may be adjusted to favor a verified, deployable result within the hackathon deadline.

## Status

Foundation initialized. Flood-model inference, data acquisition, spatial validation, network analysis, UI, and deployment are not yet verified.

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) for milestones.

## Responsible use and provenance

- Record the dataset, tile/scene identifier, acquisition time when available, model/checkpoint source, preprocessing, threshold, and model license.
- Do not describe a prediction as a confirmed bridge/road failure unless supported by appropriate evidence.
- Treat road closures and isolation as inferred only when the underlying road data and analysis justify that conclusion.
- Never fabricate performance metrics or claim field validation that has not occurred.
- AI coding assistants are used during development; the implementation and results will be checked and described transparently.

## Development

Setup and run instructions will be added after the first reproducible inference path has been verified. Large datasets, imagery, and model weights should remain outside Git; see `.gitignore`.
