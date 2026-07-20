# Assembly bench env server — Docker backend for the HUD SDK.
#
# Self-contained: Isaac Sim base + Isaac Lab (Arena's pinned checkout) +
# isaaclab_arena runtime + hud SDK + this bench. Deliberately excludes Arena's
# dev stack (Isaac-GR00T submodule, isaacteleop, gh/pipx tooling).
#
# Build (context = the workspace root holding hud-python/, assembly_bench/,
# assembly_bench/submodules/IsaacLab-Arena/):
#
#   docker build -f assembly_bench/docker/Dockerfile -t hud-assembly-env .
#
# GPU REQUIRED — Isaac renders with RTX (no CPU fallback). Run with:
#   docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -p 8765:8765 hud-assembly-env

ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0-dev2
FROM ${BASE_IMAGE}

USER root

# Hide conflicting Vulkan files (upstream Arena recipe does the same).
RUN if [ -e "/usr/share/vulkan" ] && [ -e "/etc/vulkan" ]; then \
      mv /usr/share/vulkan /usr/share/vulkan_hidden; \
    fi
RUN apt-get update && apt-get install -y git cmake ffmpeg && rm -rf /var/lib/apt/lists/*

# Isaac's kit interpreter — the only python that can import isaaclab + the kit
# runtime, so every pip/serve step goes through it.
ENV ISAAC_PY=/isaac-sim/python.sh
WORKDIR /workspace

# Isaac Lab, at the commit Arena pins.
COPY assembly_bench/submodules/IsaacLab-Arena/submodules/IsaacLab /workspace/submodules/IsaacLab
ENV ISAACLAB_PATH=/workspace/submodules/IsaacLab \
    TERM=xterm
# Install each extension with deps but WITHOUT extras — the extras are the RL
# frameworks (robomimic and friends) that upstream's `isaaclab.sh -i` pulls in
# and that this env never uses. rsl-rl is the one exception: Arena's asset
# registry imports isaaclab_rl.rsl_rl at registration time.
RUN ln -s /isaac-sim/ ${ISAACLAB_PATH}/_isaac_sim \
    && for DIR in ${ISAACLAB_PATH}/source/isaaclab*/; do \
         ${ISAAC_PY} -m pip install --no-cache-dir -e "$DIR" || exit 1; done \
    && ${ISAAC_PY} -m pip install --no-cache-dir "rsl-rl-lib==5.0.1" \
    && ${ISAAC_PY} -m pip install --force-reinstall daqp==0.8.5 \
    && chmod 777 -R /isaac-sim/kit/ && chmod a+x /isaac-sim

# Arena runtime. The embodiment registry imports every embodiment eagerly, so
# the sibling source packages (g1, ...) and pin-pink must be present even
# though this bench only uses DROID.
COPY assembly_bench/submodules/IsaacLab-Arena/setup.py assembly_bench/submodules/IsaacLab-Arena/extension.toml /workspace/
COPY assembly_bench/submodules/IsaacLab-Arena/isaaclab_arena /workspace/isaaclab_arena
COPY assembly_bench/submodules/IsaacLab-Arena/isaaclab_arena_environments /workspace/isaaclab_arena_environments
COPY assembly_bench/submodules/IsaacLab-Arena/isaaclab_arena_examples /workspace/isaaclab_arena_examples
COPY assembly_bench/submodules/IsaacLab-Arena/isaaclab_arena_g1 /workspace/isaaclab_arena_g1
COPY assembly_bench/submodules/IsaacLab-Arena/isaaclab_arena_gr00t /workspace/isaaclab_arena_gr00t
COPY assembly_bench/submodules/IsaacLab-Arena/isaaclab_arena_openpi /workspace/isaaclab_arena_openpi
RUN ${ISAAC_PY} -m pip install --no-cache-dir "pin-pink==3.1.0" \
    && ${ISAAC_PY} -m pip install --no-cache-dir -e /workspace/

# The hud SDK from local source. Kit's python dedupes shared packages as
# symlink farms across its extension prebundles, and a naive upgrade deletes
# the files those symlinks point at. So: shadow-install (--ignore-installed)
# the few packages hud genuinely needs newer, then install hud against a
# constraints freeze of everything else (pip can add, never upgrade/remove).
COPY hud-python /app/hud-python
# openai is excluded from the freeze: kit ships 2.46 while hud pins 2.44, and
# the env server never imports kit's openai path.
RUN ${ISAAC_PY} -m pip install --no-cache-dir --ignore-installed \
        "packaging>=24.0" "pydantic>=2.11.7" "pyperclip>=1.9.0" "uvicorn>=0.35" "websockets>=15.0.1" \
        "openai==2.44.0" \
    && ${ISAAC_PY} -m pip list --format=freeze --exclude-editable 2>/dev/null \
        | grep -v -E "\+|@ file|prompt.toolkit|^(packaging|pydantic|pyperclip|uvicorn|websockets|openai)==" \
        > /tmp/kit-constraints.txt \
    && ${ISAAC_PY} -m pip install --no-cache-dir -c /tmp/kit-constraints.txt \
        "/app/hud-python" msgpack \
    # PyAV is the only extra dependency HUD video streaming needs. Install it
    # bare so openpi-client's numpy<2 pin cannot downgrade Isaac's numpy 2.x.
    && ${ISAAC_PY} -m pip install --no-cache-dir --no-deps "av>=12" \
    # openpi-client pins numpy<2 but only needs its msgpack codec — install
    # bare and keep kit's numpy.
    && ${ISAAC_PY} -m pip install --no-cache-dir --no-deps "openpi-client>=0.1.2" \
    && ${ISAAC_PY} -c "from openpi_client import msgpack_numpy; print('openpi/0 codec OK')"

# The bench: env server + the assembly_bench package (parts, HDRIs, variants).
COPY assembly_bench /app/assembly_bench
RUN ${ISAAC_PY} -m pip install --no-deps -e /app/assembly_bench/assembly_bench

ENV OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    PYTHONUNBUFFERED=1

# The control channel is the only port to publish (8765 = DockerRuntime's
# default); the robot WebSocket rides its capability tunnel.
EXPOSE 8765

WORKDIR /app/assembly_bench
# The base image's entrypoint launches kit directly and would eat this CMD.
ENTRYPOINT []
CMD ["/isaac-sim/python.sh", "-m", "hud.environment.server", \
     "env.py", "--host", "0.0.0.0", "--port", "8765"]
