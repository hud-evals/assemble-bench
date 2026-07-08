# Isaac Sim won't boot on Modal: what happens and why

Prepared 2026-07-07 ·

## The short version

We tried to run NVIDIA Isaac Sim (a robotics simulator) in a Modal GPU sandbox.
It won't boot. After isolating the failure, the picture is:

- Ordinary GPU **compute** (PyTorch etc.) works fine in the sandbox.
- Ordinary GPU **graphics** (Vulkan sees the real GPU) works fine too.
- What fails is the **bridge between the two**: Isaac Sim renders images with
the graphics side and then hands those images directly to the compute side
(that is how a simulator produces camera tensors for a robot policy). That
hand-off is the one thing the sandbox doesn't support, and Isaac Sim cannot
run without it.

The same container image boots and serves correctly under plain Docker on our
own GPU machine, so this is not an image or configuration problem on our side.

## Why the sandbox behaves this way (AI guess)

Fable said: 

Modal sandboxes don't give programs raw access to the GPU driver. They run inside gVisor, a security layer from Google that sits between the program and the real driver and only forwards the driver operations it explicitly knows about. Its list of known operations covers compute workloads well and basic graphics too - but the compute<->graphics hand-off (sharing GPU memory between Vulkan and CUDA, and NVIDIA's ray-tracing/denoising APIs on top of it) uses driver operations that are not on the list yet. Anything not on the list fails with the generic error "operation not supported". That is exactly the error Isaac Sim prints, seconds into boot, from precisely those components.

## What we tested (all inside one Modal sandbox, L40S GPU, driver 580.95.05)


| Test                             | What it proves                                       | Result   |
| -------------------------------- | ---------------------------------------------------- | -------- |
| PyTorch matmul on the GPU        | plain compute works                                  | pass     |
| PyTorch with expandable segments | advanced GPU memory management works                 | pass     |
| `vulkaninfo`                     | graphics sees the real GPU (not a software fallback) | pass     |
| Isaac Sim boot                   | needs the compute<->graphics hand-off                | **fail** |


The failing boot prints, ~4 seconds in and repeating forever:

```
[Error] [carb.cudainterop.plugin] CUDA error 801: cudaErrorNotSupported - operation not supported)
[Error] [carb.cudainterop.plugin] Failed to set the CUDA device: 0
```

("cudainterop" is Isaac's compute<->graphics bridge.) Later the ray-tracing
denoiser and the physics-to-tensor bridge fail with the same error 801:

```
[Error] [rtx.optixdenoising.plugin] CUDA error 801: cudaErrorNotSupported
[Error] [omni.physx.tensors.plugin] CUDA error: operation not supported
```

## Did we just use Modal wrong?

We checked at least. The only user-side switch for extra GPU functionality is the `NVIDIA_DRIVER_CAPABILITIES` environment variable; we set it to `all`, and it
took effect (that's why Vulkan works). The deeper switch lives in gVisor itself
and only Modal can set it - but our Vulkan result suggests they already do.
One more data point: Modal's own Blender rendering example uses the plain-CUDA
render path rather than OptiX (NVIDIA's much faster ray-tracing path) - which
is what you'd choose if OptiX doesn't work on the platform.

## Reproducing it (no code of ours needed)

On a Modal sandbox with any GPU, using NVIDIA's stock Isaac Sim image:

```python
image = modal.Image.from_registry("nvcr.io/nvidia/isaac-sim:6.0.0-dev2") \
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "OMNI_KIT_ACCEPT_EULA": "YES"}).entrypoint([])
sb = modal.Sandbox.create(
    "/isaac-sim/python.sh", "-c",
    "from isaacsim.simulation_app import SimulationApp; SimulationApp({'headless': True})",
    app=app, image=image, gpu="L40S", timeout=600)
```

Watch the logs for `carb.cudainterop ... CUDA error 801` within seconds. We ran
exactly this: sandbox `sb-5rV7AygqI3AWmEK4CMD5hw` (2026-07-07 ~21:37 UTC) shows
the error 11 s into boot on the stock NVIDIA image, no third-party code at all.

Other failing sandbox IDs (our full image, same signature):
`sb-SAlMuDSCpfvMUSSshySvUJ`, `sb-OJOTz6Z7rQQHh9T6YOvZ9O`.

One footnote from our isolation runs: a generic `nvidia/cuda` image only sees
the CPU software renderer in `vulkaninfo` (llvmpipe) because it ships no NVIDIA
Vulkan ICD file; the Isaac Sim image ships one and sees the real GPU. So Vulkan
does work in Modal sandboxes when the image carries the ICD - the hand-off
layer is the only piece that fails.

## Question

Is support planned for the CUDA<->graphics hand-off (memory sharing between Vulkan and CUDA, OptiX) in GPU sandboxes? This is the one missing piece for running Omniverse/Isaac Sim - and likely any simulator or renderer that feeds frames straight into ML code - happy to run any instrumented build or diagnostic on our image to enumerate the exact unsupported operations.