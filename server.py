"""
LMB Weight Prediction Server
Fish Nutrigenomics & AI Lab | Dr. Yathish Ramena, Director
University of Arkansas at Pine Bluff

Two-model architecture:
  - Shrimp / Prawn  → local YOLO seg model (weights.pt)
  - Largemouth Bass → local YOLOv11s-seg model (lmb_weights.pt)
                      mask → fixed px/cm → allometric curve → weight

Version 4.0 — LMB weight prediction fully integrated
"""

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import tempfile
import os
import io
import logging
import requests
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("aquavision")

# ─────────────────────────────────────────────────────────────────────────────
# LMB WEIGHT PREDICTION CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Allometric curve — fitted on 51 LMB (R²=0.946, RMSE=1.75g)
LMB_ALLOMETRIC_A  = 0.007225
LMB_ALLOMETRIC_B  = 3.1607

# Scale calibration — derived from IMG_3854 (14.0cm fish = 1800px)
LMB_PX_PER_CM     = 128.57

# Fin correction — mask minAreaRect → true total length
LMB_FIN_CORRECTION = 0.954

# Training data range — warn outside this
LMB_MIN_LENGTH_CM = 11.9
LMB_MAX_LENGTH_CM = 16.6

# Confidence threshold
LMB_CONF_THRESHOLD = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA CALIBRATION — auto-detect px/cm by image width
# Ultra Wide 13mm (3024px wide) → 128.57 px/cm
# Main Camera 24mm (4284px wide) → 198.48 px/cm
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_CALIBRATION = {
    3024: 128.57,   # iPhone Ultra Wide 13mm, 12MP — derived IMG_3854 (14.0cm=1800px)
    4284: 198.48,   # iPhone Main Camera 24mm, 24MP — derived from known fish (14.5cm)
}

def get_px_per_cm(img_width: int) -> float:
    """Return px/cm calibration constant for given image width."""
    return CAMERA_CALIBRATION.get(img_width, LMB_PX_PER_CM)  # fallback = ultra wide

# ─────────────────────────────────────────────────────────────────────────────
# SHRIMP MODEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SHRIMP_MODEL_PATH  = "weights.pt"
SHRIMP_PIXELS_PER_MM = 6.5
SHRIMP_CONF_THRESHOLD = 0.40
MASK_ALPHA = 0.4

# ─────────────────────────────────────────────────────────────────────────────
# SPECIES CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SPECIES_CONFIG = {
    "vannamei": {
        "display_name": "Pacific White Shrimp",
        "scientific_name": "Litopenaeus vannamei",
        "weight_a": 8.54e-6, "weight_b": 2.997,
        "color": (0, 255, 127),
        "min_harvest_mm": 100, "optimal_harvest_mm": 130,
        "use_lmb": False,
    },
    "monodon": {
        "display_name": "Tiger Shrimp",
        "scientific_name": "Penaeus monodon",
        "weight_a": 7.2e-6, "weight_b": 3.05,
        "color": (255, 165, 0),
        "min_harvest_mm": 120, "optimal_harvest_mm": 150,
        "use_lmb": False,
    },
    "bass": {
        "display_name": "Largemouth Bass",
        "scientific_name": "Micropterus salmoides",
        "color": (100, 149, 237),
        "min_harvest_mm": 250, "optimal_harvest_mm": 350,
        "use_lmb": True,
    },
    "prawn": {
        "display_name": "Giant River Prawn",
        "scientific_name": "Macrobrachium rosenbergii",
        "weight_a": 6.8e-6, "weight_b": 3.08,
        "color": (147, 112, 219),
        "min_harvest_mm": 150, "optimal_harvest_mm": 200,
        "use_lmb": False,
    },
}

def get_species_config(key: str) -> dict:
    return SPECIES_CONFIG.get(key, SPECIES_CONFIG["vannamei"])

# ─────────────────────────────────────────────────────────────────────────────
# ROBOFLOW — hosted bass model, used when lmb_weights.pt is unavailable
# ─────────────────────────────────────────────────────────────────────────────

ROBOFLOW_API_KEY        = os.environ.get("ROBOFLOW_API_KEY", "tya4HWqSPsfoQAmR03ES")
ROBOFLOW_MODEL_ENDPOINT = os.environ.get(
    "ROBOFLOW_MODEL_ENDPOINT",
    "https://serverless.roboflow.com/bass-fish-detection-06gec/1",
)
ROBOFLOW_CONFIDENCE = 40  # Roboflow scores 0–100

# ─────────────────────────────────────────────────────────────────────────────
# ASSET PROVISIONING — fetch weights from Google Drive at import
#
# startup.py does this too (it is the Docker CMD), but doing it here as well
# means `uvicorn server:app` works on a bare host — run-local.sh, render.yaml,
# and `python -m uvicorn` all skip startup.py.
# ─────────────────────────────────────────────────────────────────────────────

GDRIVE_ASSETS = [
    {"file_id": "1IDKFEPPDOq7M1eMUogNH4CShipR2K794", "dest": "lmb_weights.pt", "min_bytes": 10_000_000},
    {"file_id": "132EAQUDsNAJr6wKntWGkhgHksLJMLR7L", "dest": "weights.pt",     "min_bytes": 10_000_000},
]

def ensure_asset(file_id: str, dest: str, min_bytes: int) -> bool:
    """Download dest from Google Drive unless a valid copy is already present."""
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        return True
    if os.path.exists(dest):
        log.warning(f"⚠️  {dest} is truncated ({os.path.getsize(dest)} bytes) — re-downloading")
        os.remove(dest)
    try:
        import gdown
    except ImportError:
        log.warning(f"⚠️  gdown not installed — cannot fetch {dest}")
        return False
    log.info(f"⬇  Fetching {dest} from Google Drive ...")
    for url in (
        f"https://drive.google.com/uc?id={file_id}",
        f"https://drive.google.com/uc?id={file_id}&confirm=t",
    ):
        try:
            gdown.download(url, dest, quiet=True, fuzzy=True)
            if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
                log.info(f"✅ {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
                return True
        except Exception as e:
            log.warning(f"   {url} failed: {e}")
    log.error(f"❌ Could not fetch {dest} — check Drive sharing ('Anyone with the link')")
    return False

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

log.info("=" * 60)
log.info("INITIALIZING AQUAVISION API v4.0")
log.info("=" * 60)

for _asset in GDRIVE_ASSETS:
    ensure_asset(**_asset)

# Shrimp model (local weights.pt)
shrimp_model = None
if os.path.exists(SHRIMP_MODEL_PATH):
    shrimp_model = YOLO(SHRIMP_MODEL_PATH)
    log.info(f"✅ Shrimp model loaded: {SHRIMP_MODEL_PATH}")
else:
    log.warning(f"⚠️  Shrimp model not found: {SHRIMP_MODEL_PATH}")

# LMB model (lmb_weights.pt — YOLOv11s-seg trained in Colab)
LMB_MODEL_PATH = "lmb_weights.pt"
lmb_model = None
if os.path.exists(LMB_MODEL_PATH):
    lmb_model = YOLO(LMB_MODEL_PATH)
    log.info(f"✅ LMB model loaded: {LMB_MODEL_PATH}")
else:
    log.warning(f"⚠️  LMB model not found: {LMB_MODEL_PATH}")
    log.warning("   Copy best.pt from Drive → rename to lmb_weights.pt")

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AquaVision API",
    description="AI-powered detection & weight estimation for aquaculture species",
    version="4.0",
)

app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — shared
# ─────────────────────────────────────────────────────────────────────────────

def encode_image_b64(img_rgb: np.ndarray) -> str:
    """Encode RGB numpy array → base64 PNG string."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""

def is_target_class(class_name: str) -> bool:
    lower = class_name.lower()
    return any(kw in lower for kw in ["shrimp", "fish", "prawn", "bass"])

def max_pairwise_distance(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    return float(np.sqrt((diff ** 2).sum(axis=2)).max())

def estimate_weight_shrimp(length_mm: float, cfg: dict) -> float:
    if length_mm <= 0:
        return 0.0
    return cfg["weight_a"] * (length_mm ** cfg["weight_b"])

def get_size_category(length_mm: float, cfg: dict) -> str:
    mn, op = cfg["min_harvest_mm"], cfg["optimal_harvest_mm"]
    if length_mm < mn * 0.7:   return "juvenile"
    elif length_mm < mn:        return "sub-harvest"
    elif length_mm < op:        return "harvestable"
    return "optimal"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — LMB specific
# ─────────────────────────────────────────────────────────────────────────────

def build_histograms(all_lengths: list, all_weights: list,
                     display_name: str, length_unit: str = "mm") -> dict:
    """Length/weight distribution PNGs (base64) — every species tab renders these."""
    histograms: dict = {}
    if not all_lengths:
        return histograms
    for metric, vals, unit, color in [
        ("length", all_lengths, length_unit, "#0066FF"),
        ("weight", all_weights, "g",         "#7B61FF"),
    ]:
        if not vals:
            continue
        try:
            fig, ax = plt.subplots(figsize=(6, 3), facecolor="#F8FAFC")
            ax.set_facecolor("#F8FAFC")
            ax.hist(vals, bins=20, color=color, edgecolor="white", alpha=0.85)
            avg = float(np.mean(vals))
            ax.axvline(avg, color="#00C48C", linestyle="--", linewidth=2,
                       label=f"Mean: {avg:.1f}{unit}")
            ax.set_xlabel(f"{metric.capitalize()} ({unit})", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title(f"{display_name} {metric.capitalize()} Distribution",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, facecolor="#F8FAFC", bbox_inches="tight")
            plt.close(fig)
            histograms[f"{metric}_histogram_base64"] = base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            log.error(f"Histogram error: {e}")
    return histograms

def predict_lmb_weight(length_cm: float) -> float:
    """Allometric power law: W = a × L^b"""
    return LMB_ALLOMETRIC_A * (length_cm ** LMB_ALLOMETRIC_B)

def contour_to_length_px(poly: np.ndarray) -> float:
    """
    Fit minimum-area rotated rect to polygon contour.
    Returns major axis length × fin correction = true total length proxy.
    """
    contour = poly.astype(np.int32).reshape(-1, 1, 2)
    rect    = cv2.minAreaRect(contour)
    raw_px  = max(rect[1])
    return raw_px * LMB_FIN_CORRECTION

def lmb_range_warning(length_cm: float) -> Optional[str]:
    if length_cm > LMB_MAX_LENGTH_CM:
        return f"Length {length_cm:.1f}cm exceeds training range (max {LMB_MAX_LENGTH_CM}cm) — prediction less reliable"
    if length_cm < LMB_MIN_LENGTH_CM:
        return f"Length {length_cm:.1f}cm below training range (min {LMB_MIN_LENGTH_CM}cm) — prediction less reliable"
    return None

def process_bass_with_roboflow(image_path: str) -> dict:
    """
    Fallback bass detector: Roboflow's hosted model. No local weights needed.

    Roboflow returns boxes + confidence only — there is no calibrated length,
    so length_cm/weight_g come back as em-dashes rather than fabricated numbers.
    Same response shape as process_lmb_image() so the UI needs no changes.
    """
    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        resp = requests.post(
            ROBOFLOW_MODEL_ENDPOINT,
            params  = {"api_key": ROBOFLOW_API_KEY, "confidence": ROBOFLOW_CONFIDENCE},
            data    = image_b64,
            headers = {"Content-Type": "application/x-www-form-urlencoded"},
            timeout = 30,
        )
        if resp.status_code != 200:
            return {"error": f"Roboflow API error {resp.status_code}: {resp.text[:200]}"}
        predictions = resp.json().get("predictions", [])
    except Exception as e:
        return {"error": f"Roboflow request failed: {e}"}

    bgr = cv2.imread(image_path)
    if bgr is None:
        return {"error": "Could not read image."}
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    color = SPECIES_CONFIG["bass"]["color"]

    fish_predictions = []
    for i, pred in enumerate(predictions):
        x, y  = pred.get("x", 0), pred.get("y", 0)
        w, h  = pred.get("width", 0), pred.get("height", 0)
        conf  = float(pred.get("confidence", 0))
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 3)
        label = f"{pred.get('class', 'bass')} {conf * 100:.0f}%"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        ty = max(th + 10, y1 - 10)
        cv2.rectangle(rgb, (x1 - 2, ty - th - 8), (x1 + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(rgb, label, (x1, ty - 2), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        fish_predictions.append({
            "fish_id"    : i + 1,
            "length_cm"  : "—",
            "weight_g"   : "—",
            "confidence" : round(conf, 3),
            "warning"    : None,
        })

    log.info(f"  Roboflow: {len(fish_predictions)} detection(s)")
    return {
        "detection_count" : len(fish_predictions),
        "fish_predictions": fish_predictions,
        "annotated_b64"   : encode_image_b64(rgb),
        "calibration"     : {
            "note"  : "Roboflow hosted model — detection + confidence only. "
                      "Length/weight needs the local LMB model (lmb_weights.pt).",
            "source": ROBOFLOW_MODEL_ENDPOINT,
        },
    }


def process_lmb_image(image_path: str, cfg: dict) -> dict:
    """
    Run YOLOv11s-seg on a fish image.
    Returns annotated image + per-fish length/weight predictions.
    """
    if lmb_model is None:
        return {"error": "LMB model not loaded. Place lmb_weights.pt in the server directory."}

    bgr = cv2.imread(image_path)
    if bgr is None:
        return {"error": "Could not read image."}

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Auto-detect calibration from image width
    img_width = bgr.shape[1]
    px_per_cm = get_px_per_cm(img_width)
    log.info(f"  Camera calibration: width={img_width}px → {px_per_cm} px/cm")
    results = lmb_model.predict(
        source   = rgb,
        conf     = LMB_CONF_THRESHOLD,
        imgsz    = 640,
        verbose  = False,
    )
    result = results[0]
    n_det  = len(result.boxes) if result.boxes is not None else 0

    # Build annotated image
    fig, ax = plt.subplots(figsize=(8, 10), dpi=100)
    ax.imshow(rgb)
    ax.axis("off")

    fish_predictions = []
    # Normalised boxes so the browser can re-draw the detections over the user's
    # own copy of the image (the reticle reveal in the console).
    detections = []
    img_h = bgr.shape[0]

    if result.masks is not None and n_det > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        for i, (poly, conf) in enumerate(
            zip(result.masks.xy, result.boxes.conf.cpu().numpy())
        ):
            if len(poly) < 3:
                continue

            # Draw segmentation mask
            patch = plt.Polygon(
                poly, fill=True, alpha=0.35,
                facecolor="lime", edgecolor="lime", linewidth=2
            )
            ax.add_patch(patch)

            # Measure length → weight
            length_px = contour_to_length_px(poly)
            length_cm = length_px / px_per_cm
            weight_g  = predict_lmb_weight(length_cm)
            warning   = lmb_range_warning(length_cm)

            centroid  = poly.mean(axis=0)
            label     = f"Fish {i+1}: {length_cm:.1f}cm → {weight_g:.1f}g  [conf={conf:.2f}]"

            ax.text(
                centroid[0], centroid[1] - 30, label,
                fontsize=10, color="white", fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="black", alpha=0.75
                ),
            )

            fish_predictions.append({
                "fish_id"    : i + 1,
                "length_px"  : round(float(length_px), 1),
                "length_cm"  : round(float(length_cm), 2),
                "weight_g"   : round(float(weight_g), 1),
                "confidence" : round(float(conf), 3),
                "warning"    : warning,
            })

            bx = xyxy[i]
            detections.append({
                "box"       : [round(float(bx[0]) / img_width, 5), round(float(bx[1]) / img_h, 5),
                               round(float(bx[2]) / img_width, 5), round(float(bx[3]) / img_h, 5)],
                "length"    : round(float(length_cm), 2),
                "weight"    : round(float(weight_g), 1),
                "confidence": round(float(conf), 3),
            })

            log.info(
                f"  Fish {i+1}: {length_px:.0f}px → {length_cm:.1f}cm "
                f"→ {weight_g:.1f}g [conf={conf:.2f}]"
                + (f" ⚠️  {warning}" if warning else "")
            )

    ax.set_title(
        f"LMB Weight Prediction  |  px/cm={px_per_cm}  |  W=a·L^b",
        fontsize=9, pad=6
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "detection_count" : n_det,
        "fish_predictions": fish_predictions,
        "detections"      : detections,
        "image_size"      : [img_width, img_h],
        "annotated_b64"   : annotated_b64,
        "calibration"     : {
            "px_per_cm"      : px_per_cm,
            "camera_width_px": img_width,
            "fin_correction" : LMB_FIN_CORRECTION,
            "formula"        : f"W = {LMB_ALLOMETRIC_A} × L^{LMB_ALLOMETRIC_B}",
            "training_range" : f"{LMB_MIN_LENGTH_CM}–{LMB_MAX_LENGTH_CM} cm",
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "ok"             : True,
        "shrimp_model"   : shrimp_model is not None,
        "lmb_model"      : lmb_model is not None,
        "version"        : "4.0",
        "timestamp"      : datetime.now().isoformat(),
        "bass_backend"   : "local LMB model" if lmb_model is not None else "Roboflow hosted",
        "roboflow_configured": bool(ROBOFLOW_API_KEY),
        "lmb_config"     : {
            "px_per_cm"       : LMB_PX_PER_CM,
            "fin_correction"  : LMB_FIN_CORRECTION,
            "allometric_a"    : LMB_ALLOMETRIC_A,
            "allometric_b"    : LMB_ALLOMETRIC_B,
            "training_range"  : f"{LMB_MIN_LENGTH_CM}–{LMB_MAX_LENGTH_CM} cm",
        },
    }

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/species")
def list_species():
    return {
        k: {
            "display_name"     : v["display_name"],
            "scientific_name"  : v["scientific_name"],
            "detection_method" : "LMB YOLOv11s-seg + allometric curve"
                                  if v.get("use_lmb") else "Local YOLO seg",
        }
        for k, v in SPECIES_CONFIG.items()
    }

# ── LMB dedicated endpoint ────────────────────────────────────────────────────

@app.post("/detect/bass")
async def detect_bass(
    files: List[UploadFile] = File(...),
):
    """
    LMB weight prediction endpoint.
    Pipeline: YOLOv11s-seg mask → px/cm calibration → allometric curve → weight_g
    """
    per_image   = []
    all_lengths = []
    all_weights = []
    total_fish  = 0

    for up in files:
        suffix = os.path.splitext(up.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await up.read())
            image_path = tmp.name

        try:
            if lmb_model is not None:
                log.info(f"🐟 LMB inference: {up.filename}")
                result = process_lmb_image(image_path, SPECIES_CONFIG["bass"])
            else:
                log.info(f"🐟 LMB model unavailable — Roboflow fallback: {up.filename}")
                result = process_bass_with_roboflow(image_path)

            if "error" in result:
                per_image.append({"filename": up.filename, "error": result["error"]})
                continue

            total_fish += result["detection_count"]
            for p in result["fish_predictions"]:
                # Roboflow gives no calibrated measurement — those come back "—"
                if isinstance(p["length_cm"], (int, float)):
                    all_lengths.append(p["length_cm"])
                if isinstance(p["weight_g"], (int, float)):
                    all_weights.append(p["weight_g"])

            per_image.append({
                "filename"         : up.filename,
                "detection_count"  : result["detection_count"],
                "fish_predictions" : result["fish_predictions"],
                "detections"       : result.get("detections", []),
                "image_size"       : result.get("image_size"),
                "calibration"      : result["calibration"],
                "annotated_image_png_base64": result["annotated_b64"],
            })

        except Exception as e:
            log.error(f"Error processing {up.filename}: {e}")
            per_image.append({"filename": up.filename, "error": str(e)})
        finally:
            try:
                os.remove(image_path)
            except Exception:
                pass

    using_lmb = lmb_model is not None
    summary   = {"total_fish": total_fish}

    # Only report measurements we actually computed — an empty run stays "—"
    # in the UI rather than reading as a confident 0.0 cm / 0.0 g.
    if all_lengths:
        summary["avg_length_cm"] = round(float(np.mean(all_lengths)), 2)
    if all_weights:
        summary["avg_weight_g"]   = round(float(np.mean(all_weights)), 1)
        summary["total_biomass_g"] = round(sum(all_weights), 1)
    if using_lmb:
        summary["formula"]   = f"W = {LMB_ALLOMETRIC_A} × L^{LMB_ALLOMETRIC_B}"
        summary["r_squared"] = 0.946
        summary["rmse_g"]    = 1.75
    else:
        summary["note"] = ("Roboflow hosted model — detection count only. "
                           "Length/weight needs lmb_weights.pt.")

    return JSONResponse({
        "timestamp"    : datetime.now().isoformat(),
        "species"      : "bass",
        "species_info" : {
            "display_name"    : "Largemouth Bass",
            "scientific_name" : "Micropterus salmoides",
        },
        "detection_method" : "YOLOv11s-seg + allometric curve (W=a·L^b)"
                             if using_lmb else "Roboflow hosted model",
        "overall_summary"  : summary,
        "histograms"       : build_histograms(all_lengths, all_weights,
                                              "Largemouth Bass", "cm"),
        "per_image" : per_image,
    })

# ── Shrimp / Prawn endpoint (unchanged) ──────────────────────────────────────

@app.post("/detect")
async def detect(
    files: List[UploadFile] = File(...),
    pixels_per_mm: Optional[float] = Query(default=None),
    species: Optional[str] = Query(default="vannamei"),
):
    """
    Shrimp / Prawn detection endpoint.
    Routes bass requests to /detect/bass automatically.
    """
    # Route bass to dedicated endpoint
    if species == "bass":
        return await detect_bass(files=files)

    if shrimp_model is None:
        return JSONResponse(
            {"error": "Shrimp model not loaded. Check weights.pt."},
            status_code=503,
        )

    calibration  = pixels_per_mm or SHRIMP_PIXELS_PER_MM
    species_cfg  = get_species_config(species)
    per_image    = []
    all_lengths  : List[float] = []
    all_weights  : List[float] = []
    overall_total = 0

    for up in files:
        suffix = os.path.splitext(up.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await up.read())
            image_path = tmp.name

        try:
            log.info(f"🦐 Shrimp inference ({species}): {up.filename}")
            results = shrimp_model(image_path, verbose=False, conf=SHRIMP_CONF_THRESHOLD)
            r = results[0]

            bgr = cv2.imread(image_path)
            if bgr is None:
                per_image.append({"filename": up.filename, "error": "Could not read image."})
                continue

            rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            overlay = rgb.copy()
            img_h, img_w = rgb.shape[:2]
            lengths_mm, weights_g, text_labels = [], [], []
            # Normalised boxes so the browser can re-draw the detections over the
            # user's own copy of the image (the reticle reveal in the console).
            detections = []

            if r.masks is not None and r.boxes is not None:
                for mask, box in zip(r.masks, r.boxes):
                    class_id   = int(box.cls[0])
                    class_name = shrimp_model.names.get(class_id, str(class_id))
                    conf       = float(box.conf[0])
                    if not is_target_class(class_name) or conf < SHRIMP_CONF_THRESHOLD:
                        continue
                    if mask.xy is None or len(mask.xy) == 0:
                        continue
                    pts = np.array(mask.xy[0], dtype=np.float32)
                    if pts.shape[0] < 2:
                        continue
                    length_mm = max_pairwise_distance(pts) / calibration
                    weight_g  = estimate_weight_shrimp(length_mm, species_cfg)
                    lengths_mm.append(float(length_mm))
                    weights_g.append(float(weight_g))
                    pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts_int], color=species_cfg["color"])
                    x1 = int(box.xyxy[0][0]); y1 = int(box.xyxy[0][1])
                    text_labels.append((x1, y1, f"{length_mm:.1f}mm | {weight_g:.2f}g"))
                    bx = [float(v) for v in box.xyxy[0]]
                    detections.append({
                        "box"       : [round(bx[0] / img_w, 5), round(bx[1] / img_h, 5),
                                       round(bx[2] / img_w, 5), round(bx[3] / img_h, 5)],
                        "length"    : round(float(length_mm), 2),
                        "weight"    : round(float(weight_g), 3),
                        "confidence": round(conf, 3),
                    })

            annotated = cv2.addWeighted(rgb, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
            for (x, y, text) in text_labels:
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
                x = max(0, x); y = max(th + 8, y - 8)
                cv2.rectangle(annotated, (x-2, y-th-8), (x+tw+4, y+4), (0,0,0), -1)
                cv2.putText(annotated, text, (x, y-2), font, 0.55, (255,255,255), 2, cv2.LINE_AA)

            n = len(lengths_mm)
            overall_total += n
            all_lengths.extend(lengths_mm)
            all_weights.extend(weights_g)
            b64 = encode_image_b64(annotated)

            per_image.append({
                "filename"       : up.filename,
                "specimen_count" : n,
                "shrimp_count"   : n,
                "average_length_mm": round(float(np.mean(lengths_mm)), 2) if n else 0.0,
                "lengths_mm"     : [round(x, 2) for x in lengths_mm],
                "weights_g"      : [round(x, 3) for x in weights_g],
                "summary": {
                    "average_length_mm" : round(float(np.mean(lengths_mm)), 2) if n else 0.0,
                    "average_weight_g"  : round(float(np.mean(weights_g)), 3) if n else 0.0,
                    "total_biomass_g"   : round(sum(weights_g), 3),
                },
                "annotated_image_png_base64": b64,
                "detections"     : detections,
                "image_size"     : [img_w, img_h],
                "detection_method": "Local YOLO",
            })

        except Exception as e:
            log.error(f"Error: {e}")
            per_image.append({"filename": up.filename, "error": str(e)})
        finally:
            try:
                os.remove(image_path)
            except Exception:
                pass

    # Histograms
    histograms = build_histograms(all_lengths, all_weights,
                                  species_cfg["display_name"], "mm")

    size_dist = {"juvenile": 0, "sub-harvest": 0, "harvestable": 0, "optimal": 0}
    for l in all_lengths:
        size_dist[get_size_category(l, species_cfg)] += 1
    total = len(all_lengths) or 1
    size_pct = {k: round(v / total * 100, 1) for k, v in size_dist.items()}

    overall_avg_len    = round(float(np.mean(all_lengths)), 2) if all_lengths else 0.0
    overall_avg_weight = round(float(np.mean(all_weights)), 3) if all_weights else 0.0
    total_biomass      = round(sum(all_weights), 3)

    return JSONResponse({
        "timestamp"        : datetime.now().isoformat(),
        "species"          : species,
        "detection_method" : "Local YOLO",
        "calibration_pixels_per_mm": calibration,
        "overall_summary"  : {
            "total_specimens"  : overall_total,
            "average_length_mm": overall_avg_len,
            "average_weight_g" : overall_avg_weight,
            "total_biomass_g"  : total_biomass,
            "total_biomass_kg" : round(total_biomass / 1000, 6),
            "size_distribution": {"counts": size_dist, "percentages": size_pct},
            "length_stats"     : {
                "min": round(min(all_lengths), 2) if all_lengths else 0,
                "max": round(max(all_lengths), 2) if all_lengths else 0,
                "std": round(float(np.std(all_lengths)), 2) if all_lengths else 0,
            },
        },
        "histograms"                : histograms,
        "per_image"                 : per_image,
        "overall_total_shrimp"      : overall_total,
        "overall_average_length_mm" : overall_avg_len,
        "histogram_png_base64"      : histograms.get("length_histogram_base64", ""),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)
