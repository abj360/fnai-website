#!/usr/bin/env bash
#
# deploy-hf.sh — push this repo to the Hugging Face Space as a Docker Space.
#
# Mirrors how the Aquaculture-vision Space is served: one origin runs both the
# landing page and the detection API, so the page's relative /detect and
# /health calls resolve. The current uapb-ai/fnai-website Space is a *static*
# Space, which serves index.html but has no Python — every analysis request
# 404s. Pushing this repo (README frontmatter declares sdk: docker) converts it.
#
# Model weights are not in the repo; startup.py fetches them from Google Drive
# on cold start, so the first boot takes a few minutes.
#
set -euo pipefail
cd "$(dirname "$0")"

SPACE="${SPACE:-uapb-ai/fnai-website}"
BRANCH="${BRANCH:-main}"

if ! command -v hf >/dev/null 2>&1; then
  echo "❌ hf CLI not found.  pip install -U huggingface_hub"
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "You are not logged in to Hugging Face. Run:"
  echo "    hf auth login        # needs a token with write access to $SPACE"
  exit 1
fi

echo "▶ Pushing to https://huggingface.co/spaces/$SPACE (branch $BRANCH)"
git remote get-url hf >/dev/null 2>&1 \
  || git remote add hf "https://huggingface.co/spaces/$SPACE"

# HF rejects non-LFS files over 10MB. Fail loudly here rather than mid-push.
oversize=$(git ls-files -z | xargs -0 -I{} sh -c \
  'test -f "{}" && s=$(stat -c%s "{}") && [ "$s" -gt 10000000 ] && echo "{}"' 2>/dev/null || true)
if [[ -n "$oversize" ]]; then
  echo "❌ Tracked files over 10MB (need git-lfs, or fetch them at runtime):"
  echo "$oversize" | sed 's/^/     /'
  exit 1
fi

git push hf "$BRANCH:main" "$@"

cat <<EOF

════════════════════════════════════════════════════════════
  Pushed. The Space rebuilds from the Dockerfile.

  Build log:  https://huggingface.co/spaces/$SPACE?logs=build
  Live:       https://$(echo "$SPACE" | tr '/' '-' | tr '[:upper:]' '[:lower:]').hf.space

  First boot downloads weights.pt (~125MB) and lmb_weights.pt (~20MB)
  from Google Drive — expect a few minutes before /health responds.

  Verify:  curl https://<space>.hf.space/health
════════════════════════════════════════════════════════════
EOF
