---
title: AquaVision Tool
emoji: 🐟
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# AquaVision — Detection Tool (UAPB Fish Nutrigenomics & AI Lab)

Companion tool to the lab's website. Upload a specimen image and the app runs the
lab-trained YOLO models to detect, measure, and estimate weight:

- **Shrimp / Prawn** (*L. vannamei*, *P. monodon*, *M. rosenbergii*) — YOLO segmentation →
  max-caliper length → allometric weight (`W = a·Lᵇ`).
- **Largemouth Bass** (*M. salmoides*) — YOLOv11s-seg mask → px/cm calibration →
  allometric curve (`W = 0.007225·L^3.1607`, R²=0.946).

Runs on **ZeroGPU**. Model weights (`weights.pt`, `lmb_weights.pt`) ship in this Space.

Main site: https://uapb-ai-fnai-website.static.hf.space
