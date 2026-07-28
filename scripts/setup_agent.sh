#!/usr/bin/env bash
# Install the *agent-side* deps for Path B (VLA eval). Use a normal Python 3.10+
# env — not the Isaac Sim one. The agent talks to the served env over TCP.
#
#   ./scripts/setup_agent.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
echo "[setup-agent] python: $($PY -c 'import sys; print(sys.executable)')"

$PY -m pip install -r requirements-agent.txt

cat <<'EOF'

[setup-agent] done.

The pi0.5 tokenizer is gated. Accept it once, then log in:
  https://huggingface.co/google/paligemma-3b-pt-224
  hf auth login

Serve the env (Isaac / Docker — see README Path B), then:
  python examples/run_eval.py --task peg_round_16mm --num-envs 4
EOF
