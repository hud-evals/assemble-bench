"""Benchmark cameras: the embodiment's own wrist camera + one frontal exterior.

The DROID embodiment already ships the right cameras (calibrated Robotiq wrist
mount, 16:9 DROID intrinsics), so this module keeps their pose/intrinsics and
only swaps the two over-shoulder exterior views for a single frontal one, and
drops both cameras' render resolution to 640x360 (see RENDER_W/H). Frames stream
at 16:9 -- model-input sizing (openpi's ``resize_with_pad`` to 224x224) is the
policy adapter's job, exactly as in the chess bench's pi0.5 eval.
"""

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.configclass import make_configclass

# DROID intrinsics (2.8 mm focal, 5.376 x 3.024 mm aperture, ~88 x 57 deg FOV),
# matching Arena's DroidCameraCfg. RENDER res is 640x360 (half the DROID-native
# 1280x720, same 16:9 FOV/intrinsics -- only pixel count changes): we store
# 320x180 (DROID-RLDS) so rendering 1280x720 was 16x wasteful and the cameras
# are the per-step cost. Recorder area-downscales 640x360 -> 320x180 (clean 2x).
RENDER_W, RENDER_H = 640, 360
_SPAWN = dict(focal_length=2.8, focus_distance=28.0,
              horizontal_aperture=5.376, vertical_aperture=3.024,
              clipping_range=(0.01, 6.0))

# Frontal exterior view (LIBERO agentview convention): close in front of the
# workspace on the robot midline, low enough that the parts read as 3D shapes,
# with the arm entering from the far side.
FRONT_CAM_EYE = (0.55, 0.0, 0.30)
FRONT_CAM_TARGET = (0.37, 0.0, 0.03)


def lookat_opengl_quat_xyzw(eye, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (x,y,z,w) aiming an OpenGL/USD-convention camera (looks down
    local -Z, +Y up) from ``eye`` to ``target``. Baked into the OffsetCfg: it is
    per-env-relative, so one offset works for all envs."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)                        # forward (view dir); camera -Z
    r = np.cross(f, up)
    r /= np.linalg.norm(r)                        # right; camera +X
    u = np.cross(r, f)                            # true up; camera +Y
    R = np.column_stack([r, u, -f])               # camera->world basis
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0]
        w = (R[k, j] - R[j, k]) / s
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        x, y, z = q
    return (float(x), float(y), float(z), float(w))


def front_camera(eye=FRONT_CAM_EYE, target=FRONT_CAM_TARGET, name="front_cam") -> CameraCfg:
    """The fixed frontal exterior camera (DROID-native 16:9)."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        offset=CameraCfg.OffsetCfg(
            pos=eye, rot=lookat_opengl_quat_xyzw(eye, target), convention="opengl"),
        update_period=0.0,
        height=RENDER_H,
        width=RENDER_W,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(**_SPAWN),
    )


def make_assembly_camera_cfg(embodiment):
    """The embodiment's camera config with its exterior views replaced by
    ``front_cam``. Wrist cameras keep the calibrated Robotiq mount pose/intrinsics
    (verified frame-for-frame against the source benchmark's demos) but their
    render resolution is dropped to match the front cam (RENDER_W/H).
    """
    fields = []
    for name in getattr(embodiment.camera_config, "__dataclass_fields__", {}):
        cam = getattr(embodiment.camera_config, name)
        if isinstance(cam, CameraCfg) and "wrist" in name:
            # Match the front cam's render res (same FOV, fewer pixels): the
            # calibrated mount/intrinsics are unchanged, only resolution drops.
            cam.height, cam.width = RENDER_H, RENDER_W
            fields.append((name, CameraCfg, cam))
    fields.append(("front_cam", CameraCfg, front_camera()))
    # Arena embodiments require camera_config to be an ArenaCameraCfg subclass
    # (get_cfg() returns tiled/untiled). make_configclass defaults to no bases.
    return make_configclass("AssemblyCameraCfg", fields, bases=(ArenaCameraCfg,))()
