---
title: AquaVision
emoji: 🐟
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AquaVision — Fish Nutrigenomics & AI Lab (UAPB)

AI-powered detection and weight estimation for aquaculture species, built by the
Fish Nutrigenomics & AI Lab at the University of Arkansas at Pine Bluff
(Dr. Yathish Ramena, Director).

Two-model architecture served by a FastAPI backend:

- **Shrimp / Prawn** → local YOLO segmentation model (`weights.pt`)
- **Largemouth Bass** → YOLOv11s-seg (`lmb_weights.pt`) → mask → px/cm calibration →
  allometric curve → weight (g)

## How it runs

This is a **Docker** Space. On cold start, [`startup.py`](startup.py) downloads the
model weights (and video assets) from Google Drive, then launches
`uvicorn server:app` on port **7860**.

### Endpoints

- `GET  /` — the AquaVision website
- `GET  /health` — model + config status
- `POST /detect/bass` — largemouth bass weight prediction
- `POST /detect?species=...` — shrimp/prawn detection
- `GET  /species` — configured species
- `GET  /docs` — interactive API docs
