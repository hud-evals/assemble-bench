#!/usr/bin/env bash
# Install Path B *agent-side* deps into a normal Python 3.12+ env (not Isaac).
# The agent talks to the served env over TCP. Runs exactly the two pip
# commands from the README:
#
#   pip install -r requirements-agent.txt
#   pip install --no-deps openpi-client==0.1.2
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
# openpi-client pins numpy<2 but only its msgpack codec is used — keep numpy 2.x.
${PY[@]} -m pip install --no-deps "openpi-client==0.1.2"

cat <<'EOF'

[setup-agent] done.

The pi0.5 tokenizer is gated. Accept it once, then log in:
  https://huggingface.co/google/paligemma-3b-pt-224
  hf auth login

Serve the env (README Path B), then:
  python examples/run_eval.py --task peg_round_16mm --num-envs 4
EOF
