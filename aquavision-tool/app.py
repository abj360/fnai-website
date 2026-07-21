"""
AquaVision — detection & weight-estimation tool (Gradio + ZeroGPU)

Companion tool to the UAPB Fish Nutrigenomics & AI Lab site. Ported from the
lab's FastAPI backend (server.py):

  • Shrimp / Prawn  → YOLO segmentation → max-caliper length → allometric weight
  • Largemouth Bass → YOLOv11s-seg mask → px/cm calibration → allometric curve

Two local model weights (weights.pt, lmb_weights.pt) ship in this Space.
"""

import io
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spaces          # ZeroGPU
import torch
import gradio as gr
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (mirrors server.py)
# ─────────────────────────────────────────────────────────────────────────────
LMB_ALLOMETRIC_A   = 0.007225
LMB_ALLOMETRIC_B   = 3.1607
LMB_PX_PER_CM      = 128.57
LMB_FIN_CORRECTION = 0.954
LMB_MIN_LENGTH_CM  = 11.9
LMB_MAX_LENGTH_CM  = 16.6
LMB_CONF_THRESHOLD = 0.25

CAMERA_CALIBRATION = {3024: 128.57, 4284: 198.48}

SHRIMP_PIXELS_PER_MM  = 6.5
SHRIMP_CONF_THRESHOLD = 0.40
MASK_ALPHA            = 0.4

SPECIES_CONFIG = {
    "vannamei": {"display_name": "Pacific White Shrimp", "scientific_name": "Litopenaeus vannamei",
                 "weight_a": 8.54e-6, "weight_b": 2.997, "color": (0, 255, 127),
                 "min_harvest_mm": 100, "optimal_harvest_mm": 130, "use_lmb": False},
    "monodon":  {"display_name": "Tiger Shrimp", "scientific_name": "Penaeus monodon",
                 "weight_a": 7.2e-6, "weight_b": 3.05, "color": (255, 165, 0),
                 "min_harvest_mm": 120, "optimal_harvest_mm": 150, "use_lmb": False},
    "bass":     {"display_name": "Largemouth Bass", "scientific_name": "Micropterus salmoides",
                 "color": (100, 149, 237), "min_harvest_mm": 250, "optimal_harvest_mm": 350,
                 "use_lmb": True},
    "prawn":    {"display_name": "Giant River Prawn", "scientific_name": "Macrobrachium rosenbergii",
                 "weight_a": 6.8e-6, "weight_b": 3.08, "color": (147, 112, 219),
                 "min_harvest_mm": 150, "optimal_harvest_mm": 200, "use_lmb": False},
}
LABEL_TO_KEY = {f"{v['display_name']} ({v['scientific_name']})": k for k, v in SPECIES_CONFIG.items()}

# ─────────────────────────────────────────────────────────────────────────────
# MODELS — loaded on CPU at startup; moved to GPU inside @spaces.GPU calls
# ─────────────────────────────────────────────────────────────────────────────
shrimp_model = YOLO("weights.pt")     if os.path.exists("weights.pt")     else None
lmb_model    = YOLO("lmb_weights.pt") if os.path.exists("lmb_weights.pt") else None

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (ported)
# ─────────────────────────────────────────────────────────────────────────────
def _device():
    return 0 if torch.cuda.is_available() else "cpu"

def max_pairwise_distance(pts):
    if pts.shape[0] < 2:
        return 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    return float(np.sqrt((diff ** 2).sum(axis=2)).max())

def estimate_weight_shrimp(length_mm, cfg):
    return cfg["weight_a"] * (length_mm ** cfg["weight_b"]) if length_mm > 0 else 0.0

def is_target_class(name):
    return any(kw in name.lower() for kw in ["shrimp", "fish", "prawn", "bass"])

def get_px_per_cm(img_width):
    return CAMERA_CALIBRATION.get(img_width, LMB_PX_PER_CM)

def predict_lmb_weight(length_cm):
    return LMB_ALLOMETRIC_A * (length_cm ** LMB_ALLOMETRIC_B)

def contour_to_length_px(poly):
    rect = cv2.minAreaRect(poly.astype(np.int32).reshape(-1, 1, 2))
    return max(rect[1]) * LMB_FIN_CORRECTION

def lmb_range_warning(length_cm):
    if length_cm > LMB_MAX_LENGTH_CM:
        return f"⚠️ {length_cm:.1f}cm > training max {LMB_MAX_LENGTH_CM}cm"
    if length_cm < LMB_MIN_LENGTH_CM:
        return f"⚠️ {length_cm:.1f}cm < training min {LMB_MIN_LENGTH_CM}cm"
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def _run_bass(image_path):
    if lmb_model is None:
        raise gr.Error("Largemouth Bass model (lmb_weights.pt) not found in this Space.")
    lmb_model.to(_device())
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise gr.Error("Could not read the uploaded image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    px_per_cm = get_px_per_cm(bgr.shape[1])

    result = lmb_model.predict(source=rgb, conf=LMB_CONF_THRESHOLD, imgsz=640,
                               device=_device(), verbose=False)[0]
    n = len(result.boxes) if result.boxes is not None else 0

    fig, ax = plt.subplots(figsize=(8, 10), dpi=100)
    ax.imshow(rgb); ax.axis("off")
    rows = []
    if result.masks is not None and n > 0:
        for i, (poly, conf) in enumerate(zip(result.masks.xy, result.boxes.conf.cpu().numpy())):
            if len(poly) < 3:
                continue
            ax.add_patch(plt.Polygon(poly, fill=True, alpha=0.35,
                                     facecolor="lime", edgecolor="lime", linewidth=2))
            length_cm = contour_to_length_px(poly) / px_per_cm
            weight_g  = predict_lmb_weight(length_cm)
            warn      = lmb_range_warning(length_cm)
            c = poly.mean(axis=0)
            ax.text(c[0], c[1] - 30, f"Fish {i+1}: {length_cm:.1f}cm → {weight_g:.1f}g",
                    fontsize=10, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.75))
            rows.append([f"Fish {i+1}", f"{length_cm:.1f}", f"{weight_g:.1f}",
                         f"{conf*100:.0f}%", warn])
    ax.set_title(f"LMB Weight Prediction  |  {px_per_cm} px/cm  |  W = {LMB_ALLOMETRIC_A}·L^{LMB_ALLOMETRIC_B}",
                 fontsize=9, pad=6)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    annotated = np.array(plt.imread(io.BytesIO(buf.getvalue())) * 255, dtype=np.uint8)[..., :3]

    lengths = [float(r[1]) for r in rows]; weights = [float(r[2]) for r in rows]
    summary = (
        f"### Largemouth Bass — *Micropterus salmoides*\n"
        f"- **Fish detected:** {n}\n"
        f"- **Avg length:** {np.mean(lengths):.1f} cm\n" if rows else
        f"### Largemouth Bass — *Micropterus salmoides*\n- **Fish detected:** {n}\n"
    )
    if rows:
        summary += (f"- **Avg weight:** {np.mean(weights):.1f} g\n"
                    f"- **Total biomass:** {sum(weights):.1f} g\n\n"
                    f"Model: YOLOv11s-seg · W = {LMB_ALLOMETRIC_A}×L^{LMB_ALLOMETRIC_B} "
                    f"(R²=0.946) · calibration {px_per_cm} px/cm")
    return annotated, summary, rows

def _run_shrimp(image_path, key):
    if shrimp_model is None:
        raise gr.Error("Shrimp/prawn model (weights.pt) not found in this Space.")
    shrimp_model.to(_device())
    cfg = SPECIES_CONFIG[key]
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise gr.Error("Could not read the uploaded image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    cal = SHRIMP_PIXELS_PER_MM

    r = shrimp_model.predict(source=rgb, conf=SHRIMP_CONF_THRESHOLD, device=_device(), verbose=False)[0]
    rows, lengths, weights, labels = [], [], [], []
    if r.masks is not None and r.boxes is not None:
        for idx, (mask, box) in enumerate(zip(r.masks, r.boxes)):
            name = shrimp_model.names.get(int(box.cls[0]), "")
            conf = float(box.conf[0])
            if not is_target_class(name) or conf < SHRIMP_CONF_THRESHOLD:
                continue
            if mask.xy is None or len(mask.xy) == 0:
                continue
            pts = np.array(mask.xy[0], dtype=np.float32)
            if pts.shape[0] < 2:
                continue
            length_mm = max_pairwise_distance(pts) / cal
            weight_g  = estimate_weight_shrimp(length_mm, cfg)
            lengths.append(length_mm); weights.append(weight_g)
            cv2.fillPoly(overlay, [pts.astype(np.int32).reshape((-1, 1, 2))], color=cfg["color"])
            x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
            labels.append((x1, y1, f"{length_mm:.1f}mm | {weight_g:.2f}g"))
            rows.append([f"#{len(rows)+1}", f"{length_mm:.1f}", f"{weight_g:.2f}", f"{conf*100:.0f}%"])

    annotated = cv2.addWeighted(rgb, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
    for (x, y, text) in labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
        x = max(0, x); y = max(th + 8, y - 8)
        cv2.rectangle(annotated, (x-2, y-th-8), (x+tw+4, y+4), (0, 0, 0), -1)
        cv2.putText(annotated, text, (x, y-2), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    n = len(lengths)
    summary = f"### {cfg['display_name']} — *{cfg['scientific_name']}*\n- **Specimens detected:** {n}\n"
    if n:
        summary += (f"- **Avg length:** {np.mean(lengths):.1f} mm\n"
                    f"- **Avg weight:** {np.mean(weights):.2f} g\n"
                    f"- **Total biomass:** {sum(weights):.2f} g\n\n"
                    f"Model: YOLO-seg · W = {cfg['weight_a']:.2e}×L^{cfg['weight_b']} · {cal} px/mm")
    return annotated, summary, rows

@spaces.GPU(duration=90)
def analyze(species_label, image_path):
    if not image_path:
        raise gr.Error("Please upload an image first.")
    key = LABEL_TO_KEY.get(species_label, "vannamei")
    if SPECIES_CONFIG[key].get("use_lmb"):
        return _run_bass(image_path)
    return _run_shrimp(image_path, key)

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
SITE_URL = "https://uapb-ai-fnai-website.static.hf.space"

with gr.Blocks(title="AquaVision Tool — UAPB", theme=gr.themes.Soft(primary_hue="amber")) as demo:
    gr.Markdown(
        f"""# 🐟 AquaVision — Detection Tool
    Computer-vision detection, measurement & weight estimation for aquaculture species,
    from the **UAPB Fish Nutrigenomics & AI Lab**. &nbsp; [← Back to the main site]({SITE_URL})
    """
    )
    with gr.Row():
        with gr.Column(scale=1):
            species = gr.Dropdown(choices=list(LABEL_TO_KEY.keys()),
                                  value=list(LABEL_TO_KEY.keys())[0], label="Species / model")
            image = gr.Image(type="filepath", label="Upload specimen image", sources=["upload"])
            run = gr.Button("Analyze", variant="primary")
            gr.Markdown("Bass uses a calibrated top-down tray photo; shrimp/prawn use a "
                        "fixed 6.5 px/mm scale. Results are estimates from lab-trained models.")
        with gr.Column(scale=1):
            out_img = gr.Image(label="Annotated result")
            out_md = gr.Markdown()
            out_tbl = gr.Dataframe(headers=["ID", "Length", "Weight", "Conf", "Note"],
                                   label="Per-specimen", wrap=True)
    run.click(analyze, inputs=[species, image], outputs=[out_img, out_md, out_tbl])

if __name__ == "__main__":
    # ssr_mode=False → classic client-rendered SPA, which embeds reliably in a
    # cross-origin iframe (gradio 5 SSR breaks the UI when embedded).
    demo.launch(ssr_mode=False)
