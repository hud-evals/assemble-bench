# Parts

```
parts/
├── used/                 ← wired by environments/assembly/scene.py
│   ├── pegs/             loose pegs + matching holes (also used as stands)
│   ├── gears/            re-centered gear meshes
│   ├── nuts/             factory_*_loose nut/bolt pairs (see below)
│   └── nist_gmc_base.usd NIST board
└── debug/                apple/bowl sanity-check (gitignored; local only)
```

**`factory_*` naming.** Those USDs are converted from NVIDIA’s
[Isaac Gym Envs Factory](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
`assets/factory` nut/bolt OBJs (same family as Isaac Lab’s Factory / FORGE
assembly tasks). The stem keeps provenance (`factory_nut_m16_loose.usd`); the
Arena registry name is still `asm_nut_m16_loose`. Pegs/gears use `gen_*` because
they were authored procedurally in this repo, not converted from Factory meshes.

Register new parts in `scene.py`, then reference them from `variants.py`.
