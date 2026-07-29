# Parts

```
parts/
├── used/                 ← NIST assembly (wired by scene.py)
│   ├── pegs/             loose pegs + matching holes (also used as stands)
│   ├── gears/            re-centered gear meshes
│   ├── nuts/             factory_*_loose nut/bolt pairs
│   └── nist_gmc_base.usd NIST board
└── debug/                smoke pick-place only (apple → bowl; not NIST)
    ├── apple_01.usd
    └── bowl.usd
```

**`factory_*` naming.** Nut/bolt USDs are converted from NVIDIA’s Isaac Gym
Envs Factory meshes. Pegs/gears use `gen_*` (authored in this repo). Arena
registry names are still `asm_*`.

**`debug/`.** Hello-world variant `--task debug` / `tasks/vla/debug.json` —
sanity-check pick-and-place, not part of the NIST matrix.

Register new parts in `scene.py`, then reference them from `variants.py`.
