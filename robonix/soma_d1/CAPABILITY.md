# soma_d1 — D1 Body Model (Soma)

Serves the three `robonix/system/soma` contracts for the Beingbeyond D1, so
downstream consumers (scene, navigation, collision monitors) can query what
this robot is instead of hardcoding a URDF path.

## The D1 is a fixed-base arm

The D1 has revolute joints only — **no wheels, no mobile chassis, no chassis
primitive**. So its footprint is a small static base hull and it cannot
navigate. A navigation consumer that inspects soma (description URDF has no
wheel joints; no chassis registered in atlas) can reject "move to X" on
embodied grounds.

## Capabilities (all gRPC, rpc)

- `robonix/system/soma/description` — D1 URDF + `{model_name, mass_kg, base_frame}`.
- `robonix/system/soma/footprint` — 2D base footprint polygon in `base_frame`
  plus inscribed / circumscribed radii.
- `robonix/system/soma/sensor_extrinsics` — static sensor mounts (default none;
  the grasp skill owns its own hand-eye calibration for now).

## Config (on_init cfg)

| key | default | description |
|-----|---------|-------------|
| urdf_path | SDK bundled D1 URDF | full URDF to serve |
| model_name | beingbeyond_d1 | friendly model name |
| base_frame | link_base | kinematic root frame |
| mass_kg | -1 | total mass (-1 = unknown) |
| footprint_xy_pts | 0.20 m square | CCW footprint vertices in base_frame |
| sensors | [] | static sensor mounts (list of {sensor_name, kind, parent_frame, child_frame, transform:{x,y,z,qx,qy,qz,qw}}) |

Runs on the `bb_d1_robonix` Python 3.10 env (see `../env_setup.sh`).
