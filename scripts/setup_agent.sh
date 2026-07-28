#!/usr/bin/env bash
# Install Path B *agent-side* deps into a normal Python 3.10+ env (not Isaac).
# The agent talks to the served env over TCP.
#
#   ./scripts/setup_agent.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
  PY=(python)
elif command -v python3 >/dev/null 2>&1; then
  PY=(python3)
else
  PY=(python)
fi

echo "[setup-agent] python: $(${PY[@]} -c 'import sys; print(sys.executable)')"
${PY[@]} -m pip install -U pip
${PY[@]} -m pip install -r requirements-agent.txt

cat <<'EOF'

[setup-agent] done.

The pi0.5 tokenizer is gated. Accept it once, then log in:
  https://huggingface.co/google/paligemma-3b-pt-224
  hf auth login

Serve the env (README Path B), then:
  python examples/run_eval.py --task peg_round_16mm --num-envs 4
EOF
