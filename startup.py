"""
startup.py — AquaVision / LMB Weight Prediction
Downloads model weights + video assets from Google Drive on cold start,
then launches the FastAPI server via uvicorn.
"""

import os
import sys
import shutil
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("startup")

# Download to /tmp (always writable), then move to /app
APP_DIR = "/app"
TMP_DIR = "/tmp/aquavision_downloads"
os.makedirs(TMP_DIR, exist_ok=True)

ASSETS = [
    {
        "file_id"  : "1IDKFEPPDOq7M1eMUogNH4CShipR2K794",
        "dest"     : "lmb_weights.pt",
        "min_bytes": 10_000_000,
        "required" : True,
    },
    {
        "file_id"  : "132EAQUDsNAJr6wKntWGkhgHksLJMLR7L",
        "dest"     : "weights.pt",
        "min_bytes": 10_000_000,
        "required" : True,
    },
    {
        "file_id"  : "1VvyL-9Ni1u3ao8Avl7vf0naGMNl4loGO",
        "dest"     : "gsla-video.mp4",
        "min_bytes": 100_000,
        "required" : False,
    },
    {
        "file_id"  : "1pvMe6mNiaIE47uNex73RBecXSjBJdAA7",
        "dest"     : "AR-research.mp4",
        "min_bytes": 100_000,
        "required" : False,
    },
    {
        "file_id"  : "1NCW0-Yo7Q_efXkwbal7tQoZcCkH9heu4",
        "dest"     : "dr-ramena.png",
        "min_bytes": 100_000,
        "required" : False,
    },
    {
        "file_id"  : "1J2bq_CUtUhImcGQCavoogP6-n_A9lLbg",
        "dest"     : "ara-logo.png",
        "min_bytes": 10_000,
        "required" : False,
    },
    {
        "file_id"  : "1L1FNJsRCjDLobaajioQ-YlzkB8WOsreb",
        "dest"     : "gsla-logo.png",
        "min_bytes": 10_000,
        "required" : False,
    },
    {
        "file_id"  : "19bAC_latHTFHg-IQWVPUL_vB2W8n-NU5",
        "dest"     : "UAPB-logo.png",
        "min_bytes": 10_000,
        "required" : False,
    },
]


def already_valid(path: str, min_bytes: int) -> bool:
    return os.path.exists(path) and os.path.getsize(path) >= min_bytes


def download(file_id: str, dest_name: str, min_bytes: int) -> bool:
    import gdown
    final_path = os.path.join(APP_DIR, dest_name)
    tmp_path   = os.path.join(TMP_DIR, dest_name)
    url = f"https://drive.google.com/uc?id={file_id}"
    log.info(f"  Downloading {dest_name} ...")
    try:
        output = gdown.download(url, tmp_path, quiet=False, fuzzy=True)
        if output and os.path.exists(tmp_path) and os.path.getsize(tmp_path) >= min_bytes:
            shutil.move(tmp_path, final_path)
            size_mb = os.path.getsize(final_path) / 1e6
            log.info(f"  ✅ {dest_name}  ({size_mb:.1f} MB)")
            return True
        # Retry with confirm token
        log.warning(f"  Retrying {dest_name} with confirm token ...")
        output = gdown.download(
            f"https://drive.google.com/uc?id={file_id}&confirm=t",
            tmp_path, quiet=False
        )
        if output and os.path.exists(tmp_path) and os.path.getsize(tmp_path) >= min_bytes:
            shutil.move(tmp_path, final_path)
            size_mb = os.path.getsize(final_path) / 1e6
            log.info(f"  ✅ {dest_name}  ({size_mb:.1f} MB)")
            return True
        log.error(f"  ❌ {dest_name} download failed or file too small")
        return False
    except Exception as e:
        log.error(f"  ❌ {dest_name} error: {e}")
        return False


def main():
    log.info("=" * 60)
    log.info("AQUAVISION STARTUP")
    log.info("=" * 60)

    try:
        import gdown  # noqa
    except ImportError:
        log.info("Installing gdown ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])

    log.info("Fetching assets from Google Drive ...")
    failed_required = []

    for asset in ASSETS:
        dest      = asset["dest"]
        file_id   = asset["file_id"]
        min_bytes = asset["min_bytes"]
        required  = asset["required"]
        final_path = os.path.join(APP_DIR, dest)

        if already_valid(final_path, min_bytes):
            size_mb = os.path.getsize(final_path) / 1e6
            log.info(f"  ⏭  {dest} already present ({size_mb:.1f} MB) — skipping")
            continue

        success = download(file_id, dest, min_bytes)
        if not success and required:
            failed_required.append(dest)

    if failed_required:
        log.error("=" * 60)
        log.error(f"FATAL: Required assets failed to download: {failed_required}")
        log.error("Check Google Drive sharing settings (must be 'Anyone with link').")
        log.error("=" * 60)
        sys.exit(1)

    log.info("All required assets ready.")
    log.info("=" * 60)
    # Cloud Run (and most PaaS) inject the listen port via $PORT; default 7860 for HF Spaces.
    port = os.environ.get("PORT", "7860")
    log.info(f"Starting AquaVision server on port {port} ...")
    os.execvp("uvicorn", [
        "uvicorn", "server:app",
        "--host", "0.0.0.0",
        "--port", port,
    ])


if __name__ == "__main__":
    main()
