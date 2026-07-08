# Issue title

`reset()` returns stale camera images when fabric is enabled — `num_rerenders_on_reset` re-renders the old transforms

# Body

**Setup:** Isaac Sim 6.0.0, manager-based env with a `CameraCfg` sensor,
`use_fabric=True` (default), and a reset event that randomizes asset poses.
Repro'd on the `arena` branch (`55df2c3439`, v3.0.0-beta); the relevant code
is identical on `main`.

## What happens

If a reset event moves assets (e.g. `reset` events calling
`write_root_pose_to_sim`), the observations returned by `reset()` contain
**camera images of the scene from before the reset**, while all state
observations are post-reset. A policy's first inference each episode sees the
wrong scene, and recorded datasets have a corrupt first frame.

Repro in one number: teleport two tabletop objects a few cm in a reset event,
then compare the image returned by `reset()` with the image after one
zero-action `step()`:

```
mean |reset_image − step_image| ≈ 5.0    # expected ≈ 0; the objects visibly
                                         # sit at their OLD positions
```

Setting `num_rerenders_on_reset` (the documented remedy) changes nothing
(diff 4.96 with 2 rerenders, 4.98 with 0).

## Why

With fabric enabled, poses written through the PhysX tensor API only reach the
renderer during a **physics step**. `reset()` never steps physics:

```python
self._reset_idx(env_ids)              # events teleport assets (PhysX tensor API)
self.scene.write_data_to_sim()
self.sim.forward()                    # does NOT sync PhysX -> fabric
for _ in range(self.cfg.num_rerenders_on_reset):
    self.sim.render()                 # re-renders the OLD fabric transforms
self.obs_buf = self.observation_manager.compute()   # image: old, state: new
```

Confirmed: ten consecutive `sim.render()` calls after such a reset still draw
the old scene; one `sim.step()` (or `use_fabric=False`) makes the reset image
correct (diff ≈ 0.3).

## Suggested fix

When RTX sensors are present, sync the PhysX→fabric transforms (what a physics
step does) before the `num_rerenders_on_reset` render loop in
`ManagerBasedEnv.reset()` / the `DirectRLEnv` equivalent — or document that
under fabric these images cannot be refreshed without stepping physics.
