"""Convert IsaacGymEnvs factory nut/bolt OBJs into benchmark-ready USDs.

Source meshes (BSD-3) live under IsaacGymEnvs assets/factory/mesh/factory_nut_bolt.
They are already in meters and use the Factory frame (bolt base z=0, nut at
factory head_h). This script:

  1. Loads the OBJ (bolt / subdiv_3x nut)
  2. Remaps Z so the pair matches variants.NUTBOLT (M16 factory head is 16 mm;
     curated Isaac Lab USD + our variants use 10 mm — compress head only so
     shank pitch is preserved)
  3. Authors the same PhysX/MDL schema as gen_nutbolt_usds.py

Writes factory_* stems so a parallel gen_* density worker is not clobbered.

    /isaac-sim/python.sh scripts/convert_factory_nutbolt_usds.py \
        --src /factory_assets/IsaacGymEnvs/assets/factory \
        --out assembly_bench/assets/parts/used/nuts \
        --sizes 16 --tolerances loose
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
import yaml
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Vt, Gf

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None

# Mirror gen_nutbolt_usds.py / variants.NUTBOLT (meters).
NUTBOLT = {
    4:  dict(head_h=0.004, shank=0.016, pitch=0.0007),
    8:  dict(head_h=0.008, shank=0.018, pitch=0.00125),
    12: dict(head_h=0.012, shank=0.020, pitch=0.00175),
    16: dict(head_h=0.010, shank=0.025, pitch=0.002),
    20: dict(head_h=0.020, shank=0.045, pitch=0.0025),
}
DENSITY = 8000.0
MDL = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
       "Assets/Isaac/5.1/NVIDIA/Materials/Base/Metals/{}.mdl")
# URDF says nut 256 / bolt 512; 512/512 is what seated in the M16 expert test.
SDF_NUT = 512
SDF_BOLT = 512


def _load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def _factory_head_h(yaml_path: Path, size: int, tol: str) -> float:
    """Bolt head height from factory_asset_info_nut_bolt.yaml."""
    info = yaml.safe_load(yaml_path.read_text())
    key = f"nut_bolt_m{size}_{tol}"
    return float(info[key][f"bolt_m{size}_{tol}"]["head_height"])


def _remap_bolt(mesh: trimesh.Trimesh, factory_hh: float, target_hh: float) -> trimesh.Trimesh:
    """Compress hex head in Z; leave shank (and thread pitch) unchanged."""
    m = mesh.copy()
    if abs(factory_hh - target_hh) < 1e-9:
        return m
    z = m.vertices[:, 2]
    head = z <= factory_hh + 1e-9
    z2 = np.where(head, z * (target_hh / factory_hh), target_hh + (z - factory_hh))
    m.vertices[:, 2] = z2
    return m


def _remap_nut(mesh: trimesh.Trimesh, factory_hh: float, target_hh: float) -> trimesh.Trimesh:
    """Translate so nut base sits at variants head_h."""
    m = mesh.copy()
    m.vertices[:, 2] = m.vertices[:, 2] + (target_hh - factory_hh)
    return m


def _add_material(stage, root, name):
    mat = UsdShade.Material.Define(stage, f"{root}/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"{root}/Looks/{name}/Shader")
    sh.CreateIdAttr("mdlMaterial")
    sh.SetSourceAsset(Sdf.AssetPath(MDL.format(name)), "mdl")
    sh.SetSourceAssetSubIdentifier(name, "mdl")
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


def _write_mesh(stage, path, mesh, color):
    """Write mesh vertices already in meters (unlike gen_* which authors mm)."""
    g = UsdGeom.Mesh.Define(stage, path)
    v = np.asarray(mesh.vertices, dtype=np.float32)
    g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    g.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype=np.int32).reshape(-1)))
    g.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, np.int32)))
    g.CreateSubdivisionSchemeAttr("none")
    g.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    g.CreateExtentAttr(Vt.Vec3fArray([
        Gf.Vec3f(*map(float, v.min(0))), Gf.Vec3f(*map(float, v.max(0)))]))
    return g


def _add_collision(prim, res):
    """SDF mesh collision. Isaac's pxr build often lacks PhysxSchema, so apply
    the API by name (required for PhysX to cook SDF instead of convexHull)."""
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("sdf")
    if PhysxSchema is not None:
        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim).CreateSdfResolutionAttr(res)
    else:
        prim.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
        prim.CreateAttribute("physxSDFMeshCollision:sdfResolution",
                             Sdf.ValueTypeNames.Int).Set(res)


def write_nut(mesh, path, res):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(asset.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(asset.GetPrim())
    UsdPhysics.MassAPI.Apply(asset.GetPrim()).CreateDensityAttr(DENSITY)
    g = _write_mesh(stage, "/Asset/collisions", mesh, (0.9, 0.7, 0.2))
    _add_collision(g.GetPrim(), res)
    mat = _add_material(stage, "/Asset", "Brass")
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()


def write_bolt(mesh, path, res):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    bolt = UsdGeom.Xform.Define(stage, "/Asset/bolt")
    UsdPhysics.RigidBodyAPI.Apply(bolt.GetPrim())
    UsdPhysics.MassAPI.Apply(bolt.GetPrim()).CreateDensityAttr(DENSITY)
    g = _write_mesh(stage, "/Asset/bolt/collisions", mesh, (0.6, 0.8, 0.9))
    _add_collision(g.GetPrim(), res)
    mat = _add_material(stage, "/Asset", "Steel_Stainless")
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    joint = UsdPhysics.FixedJoint.Define(stage, "/Asset/FixedJoint")
    joint.CreateBody1Rel().SetTargets([bolt.GetPath()])
    UsdPhysics.ArticulationRootAPI.Apply(joint.GetPrim())
    stage.GetRootLayer().Save()


def convert_one(src: Path, out: Path, size: int, tol: str, nut_res: int, bolt_res: int):
    mesh_dir = src / "mesh" / "factory_nut_bolt"
    yaml_path = src / "yaml" / "factory_asset_info_nut_bolt.yaml"
    bolt_obj = mesh_dir / f"factory_bolt_m{size}_{tol}.obj"
    nut_obj = mesh_dir / f"factory_nut_m{size}_{tol}_subdiv_3x.obj"
    if not bolt_obj.is_file() or not nut_obj.is_file():
        raise FileNotFoundError(f"missing OBJ for M{size} {tol}: {bolt_obj} / {nut_obj}")

    target = NUTBOLT[size]
    factory_hh = _factory_head_h(yaml_path, size, tol)
    bolt = _remap_bolt(_load_mesh(bolt_obj), factory_hh, target["head_h"])
    nut = _remap_nut(_load_mesh(nut_obj), factory_hh, target["head_h"])

    bolt_path = out / f"factory_bolt_m{size}_{tol}.usd"
    nut_path = out / f"factory_nut_m{size}_{tol}.usd"
    write_bolt(bolt, bolt_path, bolt_res)
    write_nut(nut, nut_path, nut_res)

    bz = bolt.vertices[:, 2]
    nz = nut.vertices[:, 2]
    tip = target["head_h"] + target["shank"]
    print(
        f"M{size} {tol}: factory_hh={factory_hh*1e3:.1f}mm -> {target['head_h']*1e3:.1f}mm  "
        f"bolt_z=[{bz.min():.4f},{bz.max():.4f}] (tip_expect={tip:.4f})  "
        f"nut_z=[{nz.min():.4f},{nz.max():.4f}]  "
        f"wt_bolt={bolt.is_watertight} wt_nut={nut.is_watertight}  "
        f"sdf nut={nut_res} bolt={bolt_res}\n"
        f"  -> {nut_path.name} / {bolt_path.name}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="IsaacGymEnvs assets/factory directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16])
    ap.add_argument("--tolerances", nargs="+", default=["loose"],
                    choices=["loose", "tight"])
    ap.add_argument("--sdf_nut", type=int, default=SDF_NUT)
    ap.add_argument("--sdf_bolt", type=int, default=SDF_BOLT)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        for tol in args.tolerances:
            convert_one(src, out, size, tol, args.sdf_nut, args.sdf_bolt)


if __name__ == "__main__":
    main()
