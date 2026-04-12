"""Closed-loop GR00T → WBC bridge.

It is a glue process that:

1) Reads observations
    - Robot state from ROS 2: `/G1Env/env_state_act` (msgpack dict over `std_msgs/ByteMultiArray`)
    - Camera RGB frames from the sim process over ZMQ (see `ComposedCameraClientSensor`)

2) Calls the GR00T policy server (network RPC)
    - Sends: {video, language, state}
    - Receives: an action dictionary with keys like `left_arm`, `right_arm`, `left_hand`, ...
      typically shaped (B=1, T=30, D).

3) Publishes a *goal* to the WBC control loop over ROS 2
    - Topic: `/ControlPolicy/upper_body_pose`
    - Payload: msgpack dict with keys like:
         - `target_upper_body_pose` (joint targets for the robot model's `upper_body` group)
         - `navigate_cmd`, `base_height_command`, `wrist_pose`
         - `target_time` (monotonic timestamp used by InterpolationPolicy)

What happens next (outside this process)
--------------------------------------

The WBC control loop (`run_g1_control_loop.py`) subscribes to `/ControlPolicy/upper_body_pose`,
updates its policy goal, computes a full-body joint target `q`, and then *writes DDS motor
commands* (e.g. `rt/lowcmd` and `rt/dex3/*/cmd`) via Unitree SDK2 publishers.

So the ROS goal topic is the *input* to the control loop, and DDS is the *actuation* output.

Important gotcha: `target_time`
-------------------------------

By default, the upper-body side of `G1DecoupledWholeBodyPolicy` uses `InterpolationPolicy`.
That policy will ignore goals that do not include a `target_time`. If you publish goals
without `target_time`, you'll see the control loop receiving goals, but the arms won't move.

This bridge therefore always publishes:

     target_time = time.monotonic() + target_time_offset_s

Using monotonic time matches the control loop's expectation (no wall-clock jumps).
"""

import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import tyro

from gr00t.policy.server_client import PolicyClient

from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,  #  /ControlPolicy/upper_body_pose
    STATE_TOPIC_NAME,    #  /G1Env/env_state_act
    DEFAULT_BASE_HEIGHT, #   0.74
    DEFAULT_NAV_CMD,     #  [0.0, 0.0, 0.0]
    DEFAULT_WRIST_POSE,  #  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
)
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.sensor.composed_camera import ComposedCameraClientSensor
from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber


def _as_bt(v: np.ndarray) -> np.ndarray:
    """Ensure (B=1,T=1,D)"""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        return v[None, None, :]
    if v.ndim == 2:
        return v[None, :, :]
    if v.ndim == 3:
        return v
    raise ValueError(f"Unsupported ndim={v.ndim} for BT wrapping")


def _as_btd(v: Any) -> np.ndarray:
    """Ensure an array is (B,T,D) for consistent indexing."""
    arr = np.asarray(v, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, None, :]
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported action ndim={arr.ndim}; expected 1..3")


def _extract_wrist_state(state_msg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract GR00T wrist state channels from the control-loop state message."""
    wrist_pose = state_msg.get("wrist_pose", None)
    if wrist_pose is None:
        return {}

    wrist_pose = np.asarray(wrist_pose, dtype=np.float32).reshape(-1)
    if wrist_pose.size != 14:
        raise RuntimeError(
            f"Expected state_msg['wrist_pose'] to have 14 values, got {int(wrist_pose.size)}."
        )

    return {
        "left_wrist_pos": wrist_pose[0:3],
        "left_wrist_abs_quat": wrist_pose[3:7],
        "right_wrist_pos": wrist_pose[7:10],
        "right_wrist_abs_quat": wrist_pose[10:14],
    }


def _abs_topic(name: str) -> str:
    """Ensure topic names are absolute.

    ROS 2 name resolution can surprise you if a node has a namespace.
    Using absolute paths ensures we are publishing/subscribing to exactly:
      - `/G1Env/env_state_act`
      - `/ControlPolicy/upper_body_pose`
    """
    return name if name.startswith("/") else f"/{name}"


@dataclass
class Cfg:
    camera_host: str = "localhost"
    camera_port: int = 5555
    camera_key: str = "ego_view"

    policy_host: str = "localhost"
    policy_port: int = 5556

    rate_hz: float = 10.0
    lang_instruction: str = "Pick up the rod."

    enable_waist: bool = False
    with_hands: bool = True
    high_elbow_pose: bool = False

    # Interpret GR00T outputs
    #
    # Important: Isaac-GR00T's Unitree G1 modality config uses:
    # - arms: RELATIVE (delta) joint actions
    # - hands: ABSOLUTE (gripper-like)
    # - waist: ABSOLUTE
    #
    # This bridge therefore interprets actions per-group by default.
    arms_action_is_delta: bool = True
    hands_action_is_delta: bool = False
    waist_action_is_delta: bool = False

    # If True, query the policy server for its modality config and infer per-group action
    # semantics (RELATIVE/DELTA -> delta, ABSOLUTE -> absolute). This removes guesswork when
    # switching models.
    auto_action_semantics_from_server: bool = True
    # Precedence rule when both CLI flags and server semantics are available.
    # Default: CLI wins (so `--no-arms-action-is-delta` does what you expect).
    cli_overrides_server_action_semantics: bool = True
    # Print the server-reported action representations at startup (useful for debugging).
    debug_print_server_action_reps: bool = False

    # Only used for delta-interpreted groups.
    # If you see mapping debug reporting |Δ| max == max_delta_rad constantly,
    # you are saturating the deltas and should increase these limits.
    delta_scale: float = 1.0
    max_delta_rad: float = 0.10

    # Per-group overrides (mainly useful for arms).
    arms_delta_scale: float | None = None
    arms_max_delta_rad: float | None = None

    exec_t_index: int = 0  # 0..29

    # InterpolationPolicy requires target_time; use monotonic time to match the control loop.
    target_time_offset_s: float = 0.10

    dry_run_print_only: bool = False

    # Debug / validation helpers
    debug_joint_mapping: bool = False
    # Print debug output every N ticks (0 or 1 means every tick).
    debug_print_every: int = 20
    # Treat |Δq| <= eps as unchanged.
    debug_eps: float = 1e-6
    # Show joint names in debug output (requires RobotModel joint->index mapping).
    debug_show_joint_names: bool = True
    # Cap the number of joint names printed per group.
    debug_max_names_per_group: int = 20
    # If True, raise an error if GR00T output D doesn't match the joint group size.
    debug_fail_on_shape_mismatch: bool = True
    # If True, raise an error when unexpected joints inside upper_body are modified.
    debug_fail_on_unexpected_joint_change: bool = False


def main(cfg: Cfg):
    # Force absolute topic names to avoid namespace resolution surprises.
    state_topic = _abs_topic(STATE_TOPIC_NAME)
    goal_topic = _abs_topic(CONTROL_GOAL_TOPIC)

    # ROSManager is a thin wrapper that creates an rclpy Node and handles shutdown.
    # We deliberately create *one* node and pass it to pub/sub helpers.
    ros_manager = ROSManager(node_name="groot_closed_loop_bridge")
    node = ros_manager.node

    try:
        # ZMQ camera client: connects to the sim loop's camera server.
        cam = ComposedCameraClientSensor(server_ip=cfg.camera_host, port=cfg.camera_port)

        # These wrappers must NOT create their own node internally.
        # They serialize/deserialize msgpack dicts into ByteMultiArray.
        state_sub = ROSMsgSubscriber(node, state_topic)
        goal_pub = ROSMsgPublisher(node, goal_topic)

        # The robot model provides joint-group indices (left_arm, right_arm, upper_body, ...).
        # `waist_location` controls whether waist DOFs are included in the upper_body group.
        waist_location = "lower_and_upper_body" if cfg.enable_waist else "lower_body"
        robot_model = instantiate_g1_robot_model(
            waist_location=waist_location,
            high_elbow_pose=cfg.high_elbow_pose,
        )

        # Build a reverse lookup so we can print human-readable joint names from DOF indices.
        # This is purely for debugging / verification.
        index_to_joint_name: Dict[int, str] = {}
        if cfg.debug_joint_mapping and cfg.debug_show_joint_names:
            try:
                for joint_name in robot_model.joint_names:
                    index_to_joint_name[int(robot_model.dof_index(joint_name))] = str(joint_name)
            except Exception:
                # If the robot model implementation differs, we can still proceed without names.
                index_to_joint_name = {}

        def _name_for_index(i: int) -> str:
            return index_to_joint_name.get(int(i), f"dof[{int(i)}]")

        # GR00T PolicyClient: RPC client to the policy server (separate process).
        policy = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
        if not policy.ping():
            raise RuntimeError(
                f"Cannot connect to GR00T policy server at {cfg.policy_host}:{cfg.policy_port}"
            )

        # Optionally infer action semantics from the server's modality config.
        # This is the authoritative source for whether a given joint-group is trained as
        # relative (delta) or absolute.
        server_action_is_delta: Dict[str, bool] = {}
        server_state_keys: set[str] = set()
        server_video_keys: set[str] = set()
        if cfg.auto_action_semantics_from_server or cfg.debug_print_server_action_reps:
            try:
                modality_cfg = policy.get_modality_config()
                state_cfg = modality_cfg.get("state", None)
                if state_cfg is not None:
                    server_state_keys = set(getattr(state_cfg, "modality_keys", []) or [])

                video_cfg = modality_cfg.get("video", None)
                if video_cfg is not None:
                    server_video_keys = set(getattr(video_cfg, "modality_keys", []) or [])

                action_cfg = modality_cfg.get("action", None)
                if action_cfg is not None:
                    keys = list(getattr(action_cfg, "modality_keys", []) or [])
                    cfgs = list(getattr(action_cfg, "action_configs", []) or [])
                    if keys and cfgs and len(keys) == len(cfgs):
                        for k, ac in zip(keys, cfgs):
                            rep = getattr(ac, "rep", None)
                            rep_val = getattr(rep, "value", str(rep))
                            rep_val = str(rep_val).lower()
                            if rep_val in ("relative", "delta"):
                                server_action_is_delta[str(k)] = True
                            elif rep_val == "absolute":
                                server_action_is_delta[str(k)] = False

                        if cfg.debug_print_server_action_reps or cfg.debug_joint_mapping:
                            pretty = {k: ("delta" if v else "absolute") for k, v in server_action_is_delta.items()}
                            print("Server action reps (inferred semantics):", pretty)
                    elif cfg.debug_print_server_action_reps or cfg.debug_joint_mapping:
                        print(
                            "Server modality config does not include action_configs (or lengths mismatch); "
                            "falling back to CLI semantics flags."
                        )
            except Exception as e:
                if cfg.debug_print_server_action_reps or cfg.debug_joint_mapping:
                    print("Warning: failed to query server modality config:", repr(e))

        rate = node.create_rate(cfg.rate_hz)
        printed_once = False

        # Cache joint indices once (they are constant for the chosen robot model).
        # These indices refer to positions in the `q` vector coming from `/G1Env/env_state_act`.
        idx = {
            "left_leg": np.asarray(robot_model.get_joint_group_indices("left_leg"), dtype=np.int64),
            "right_leg": np.asarray(robot_model.get_joint_group_indices("right_leg"), dtype=np.int64),
            "waist": np.asarray(robot_model.get_joint_group_indices("waist"), dtype=np.int64),
            "left_arm": np.asarray(robot_model.get_joint_group_indices("left_arm"), dtype=np.int64),
            "right_arm": np.asarray(robot_model.get_joint_group_indices("right_arm"), dtype=np.int64),
            "left_hand": np.asarray(robot_model.get_joint_group_indices("left_hand"), dtype=np.int64),
            "right_hand": np.asarray(robot_model.get_joint_group_indices("right_hand"), dtype=np.int64),
            "upper_body": np.asarray(robot_model.get_joint_group_indices("upper_body"), dtype=np.int64),
        }

        if cfg.debug_joint_mapping:
            print("=== Joint-group mapping (indices) ===")
            for g in (
                "left_arm",
                "right_arm",
                "waist",
                "left_hand",
                "right_hand",
                "upper_body",
            ):
                arr = idx[g]
                print(f"{g}: size={arr.size} min={int(arr.min())} max={int(arr.max())}")
                if cfg.debug_show_joint_names and index_to_joint_name:
                    names = [_name_for_index(j) for j in arr.tolist()]
                    # Keep it readable: show up to N names.
                    max_n = int(max(1, cfg.debug_max_names_per_group))
                    head = names[:max_n]
                    suffix = " ..." if len(names) > max_n else ""
                    print(f"  names: {head}{suffix}")

        # Which groups we will send to the policy server.
        # (We include legs + waist so the policy can be conditioned on whole-body posture.)
        state_groups = ["left_leg", "right_leg", "waist", "left_arm", "right_arm"]
        if cfg.with_hands:
            state_groups += ["left_hand", "right_hand"]

        tick = 0

        def _group_action_is_delta(group_name: str) -> bool:
            # 1) If requested, apply CLI semantics first.
            if cfg.cli_overrides_server_action_semantics:
                if group_name in ("left_arm", "right_arm"):
                    return bool(cfg.arms_action_is_delta)
                if group_name in ("left_hand", "right_hand"):
                    return bool(cfg.hands_action_is_delta)
                if group_name == "waist":
                    return bool(cfg.waist_action_is_delta)

            # 2) Otherwise, use server semantics when available.
            if cfg.auto_action_semantics_from_server and group_name in server_action_is_delta:
                return bool(server_action_is_delta[group_name])

            # 3) Fallbacks (covers unknown groups, or when server didn't provide a rep).
            if group_name in ("left_arm", "right_arm"):
                return bool(cfg.arms_action_is_delta)
            if group_name in ("left_hand", "right_hand"):
                return bool(cfg.hands_action_is_delta)
            if group_name == "waist":
                return bool(cfg.waist_action_is_delta)
            # Fallback: treat unknown groups as delta to be conservative.
            return True

        def _delta_params_for_group(group_name: str) -> tuple[float, float]:
            """Return (scale, max_delta_rad) for a delta-interpreted joint group."""
            if group_name in ("left_arm", "right_arm"):
                scale = float(cfg.arms_delta_scale) if cfg.arms_delta_scale is not None else float(cfg.delta_scale)
                max_d = float(cfg.arms_max_delta_rad) if cfg.arms_max_delta_rad is not None else float(cfg.max_delta_rad)
                return scale, max_d
            return float(cfg.delta_scale), float(cfg.max_delta_rad)

        while ros_manager.ok():
            # ==============================
            # 1) Read latest robot state
            # ==============================
            state_msg = state_sub.get_msg()
            if state_msg is None:
                print("No ROS state yet on", state_topic)
                rate.sleep()
                continue

            # ==============================
            # 2) Read latest camera frame
            # ==============================
            img_msg = cam.read()
            if img_msg is None:
                print("No camera frame yet from", cfg.camera_host, cfg.camera_port)
                rate.sleep()
                continue

            # The state message is a dict. The minimum required field here is `q`.
            # (The control loop publishes other fields too, but we only need q to
            # build per-joint groups for conditioning and to apply deltas.)
            if "q" not in state_msg:
                print("State msg missing 'q'. Keys:", list(state_msg.keys()))
                rate.sleep()
                continue

            rgb = img_msg.get("images", {}).get(cfg.camera_key, None)
            if rgb is None:
                print(
                    "Camera msg missing key",
                    cfg.camera_key,
                    "keys:",
                    list(img_msg.get("images", {}).keys()),
                )
                rate.sleep()
                continue

            q = np.asarray(state_msg["q"], dtype=np.float32)
            nq = q.shape[0]

            # Safety checks: all indices must be in-range.
            # If these fail, your robot model mapping doesn't match the env state's q ordering.
            for k, arr in idx.items():
                if arr.size == 0:
                    raise RuntimeError(f"Joint group '{k}' is empty (robot_model mapping issue).")
                if int(arr.max()) >= nq:
                    raise RuntimeError(
                        f"Joint group '{k}' has index {int(arr.max())} but len(q)={nq}."
                    )

            # Compute which groups we will APPLY to upper_body this tick.
            # These are the groups we intend to command via `/ControlPolicy/upper_body_pose`.
            apply_groups = ["left_arm", "right_arm"]
            if cfg.enable_waist:
                apply_groups.append("waist")
            if cfg.with_hands:
                apply_groups += ["left_hand", "right_hand"]

            # ==============================
            # 3) Build GR00T observation
            # ==============================
            # GR00T expects batched, time-sequenced tensors. For closed-loop control we send a
            # single timestep (T=1) and batch size 1.
            wrist_state = _extract_wrist_state(state_msg)
            state_obs = {g: _as_bt(q[idx[g]]) for g in state_groups}
            state_obs.update({k: _as_bt(v) for k, v in wrist_state.items()})
            observation: Dict[str, Any] = {
                "video": {cfg.camera_key: rgb[None, None, ...].astype(np.uint8, copy=False)},
                "language": {"annotation.human.task_description": [[cfg.lang_instruction]]},
                "state": state_obs,
            }

            if not printed_once:
                # Sanity-check that upper_body actually contains the joints we plan to write into it.
                uset = set(map(int, idx["upper_body"].tolist()))
                missing_from_upper = []
                for g in apply_groups:
                    gset = set(map(int, idx[g].tolist()))
                    if not gset.issubset(uset):
                        missing_from_upper.append(g)

                if missing_from_upper:
                    print(
                        "Warning: these groups are not fully contained in upper_body:",
                        missing_from_upper,
                    )

                print("len(q) =", nq)
                print("State groups sent:", sorted(observation["state"].keys()))
                if server_state_keys:
                    missing_state = sorted(server_state_keys - set(observation["state"].keys()))
                    extra_state = sorted(set(observation["state"].keys()) - server_state_keys)
                    print("Server state keys:", sorted(server_state_keys))
                    if missing_state:
                        print("WARNING: missing state keys for server modality:", missing_state)
                    if extra_state:
                        print("Note: extra local state keys not declared by server modality:", extra_state)
                if server_video_keys:
                    missing_video = sorted(server_video_keys - set(observation["video"].keys()))
                    extra_video = sorted(set(observation["video"].keys()) - server_video_keys)
                    print("Server video keys:", sorted(server_video_keys))
                    if missing_video:
                        print("WARNING: missing video keys for server modality:", missing_video)
                    if extra_video:
                        print("Note: extra local video keys not declared by server modality:", extra_video)
                print("Calling policy server…")

            # ==============================
            # 4) GR00T inference
            # ==============================
            # The server returns a dict of per-group action sequences, usually shape (1, 30, D).
            action_dict, info = policy.get_action(observation)

            if not printed_once:
                print("GR00T action_dict keys:", list(action_dict.keys()))
                for k, v in action_dict.items():
                    if hasattr(v, "shape"):
                        print(f"  {k}: {v.shape}")
                printed_once = True

            # Choose which timestep to execute from the returned horizon.
            # Commonly, t=0 gives the most reactive behavior.
            t = int(cfg.exec_t_index)

            def _get_action_key(base: str) -> str:
                """Resolve action key names across different GR00T heads.

                Some models return keys like "left_arm" while others return
                "action.left_arm".
                """
                if base in action_dict:
                    return base
                prefixed = f"action.{base}"
                if prefixed in action_dict:
                    return prefixed
                raise KeyError(
                    f"Policy response missing expected key '{base}' or '{prefixed}'. "
                    f"Available keys: {list(action_dict.keys())}"
                )

            def step_vec(key: str) -> np.ndarray:
                # Normalize whatever comes back from GR00T into (B,T,D), then pick (B=0, t, :).
                arr = _as_btd(action_dict[_get_action_key(key)])
                t_clamped = int(np.clip(t, 0, arr.shape[1] - 1))
                return arr[0, t_clamped, :]

            # ==============================
            # 5) Convert GR00T outputs into joint targets
            # ==============================
            # This bridge supports two action conventions:
            #  - delta actions: GR00T outputs Δq per group
            #  - absolute actions: GR00T outputs desired q per group
            #
            # Action semantics are decided per joint-group (arms vs hands vs waist).
            targets_group: Dict[str, np.ndarray] = {}

            for g in apply_groups:
                cur = q[idx[g]].astype(np.float32, copy=False)
                u = step_vec(g)

                # Validate that GR00T's D matches the expected DOF count for this joint group.
                if u.shape[0] != cur.shape[0]:
                    msg = (
                        f"Shape mismatch for group '{g}': got GR00T D={int(u.shape[0])} "
                        f"but expected {int(cur.shape[0])} (len(idx[{g}])={int(idx[g].size)}). "
                        "This usually means the action head ordering (or robot model) does not match."
                    )
                    if cfg.debug_fail_on_shape_mismatch:
                        raise RuntimeError(msg)
                    print("WARNING:", msg)

                if _group_action_is_delta(g):
                    scale, max_d = _delta_params_for_group(g)
                    unclipped = u * scale
                    delta = np.clip(unclipped, -max_d, max_d)
                    tgt = cur + delta

                    if cfg.debug_joint_mapping and tick % int(max(1, cfg.debug_print_every)) == 0:
                        # Report how much we're clipping; constant clipping often looks like "snapping" motions.
                        clip_eps = 1e-12
                        clip_rate = float(np.mean(np.abs(unclipped) > (max_d + clip_eps)))
                        u_abs_max = float(np.max(np.abs(u))) if u.size else 0.0
                        unclipped_abs_max = float(np.max(np.abs(unclipped))) if unclipped.size else 0.0
                        print(
                            f"  {g} delta params: scale={scale:.3f} max_delta_rad={max_d:.3f} "
                            f"|u|max={u_abs_max:.3f} |scale*u|max={unclipped_abs_max:.3f} clip_rate={clip_rate:.2f}"
                        )
                else:
                    tgt = u
                targets_group[g] = tgt

            if cfg.debug_joint_mapping and tick % int(max(1, cfg.debug_print_every)) == 0:
                semantics = {g: ("delta" if _group_action_is_delta(g) else "absolute") for g in apply_groups}
                print("action semantics:", semantics)

            # Assemble a full `upper_body` target vector.
            # This is what the control loop's upper-body policy consumes.
            upper_idx = idx["upper_body"]
            target_upper = q[upper_idx].astype(np.float32, copy=True)

            # Map absolute joint indices -> position in upper_idx
            # (upper_idx is a subset of q indices; we need to write into that local ordering.)
            pos_map = {int(j): i for i, j in enumerate(upper_idx.tolist())}

            def write_group_into_upper(group_name: str):
                g_idx = idx[group_name]
                tgt = targets_group[group_name]
                for k, j in enumerate(g_idx.tolist()):
                    j = int(j)
                    if j in pos_map:
                        target_upper[pos_map[j]] = tgt[k]

            if cfg.enable_waist:
                write_group_into_upper("waist")
            write_group_into_upper("left_arm")
            write_group_into_upper("right_arm")
            if cfg.with_hands:
                write_group_into_upper("left_hand")
                write_group_into_upper("right_hand")

            # ==============================
            # 6) Debug: validate the mapping we are about to publish
            # ==============================
            if cfg.debug_joint_mapping:
                # Which absolute indices we *intend* to modify inside upper_body.
                intended_abs = set()
                for g in apply_groups:
                    intended_abs.update(map(int, idx[g].tolist()))

                # Report if any commanded joints are not present in upper_body (they would be skipped).
                upper_abs = set(map(int, upper_idx.tolist()))
                missing_by_group = {}
                for g in apply_groups:
                    missing = sorted(set(map(int, idx[g].tolist())) - upper_abs)
                    if missing:
                        missing_by_group[g] = missing

                if missing_by_group and tick % int(max(1, cfg.debug_print_every)) == 0:
                    print("WARNING: some commanded joints are missing from upper_body and will not be written")
                    for g, missing in missing_by_group.items():
                        if cfg.debug_show_joint_names and index_to_joint_name:
                            names = [_name_for_index(i) for i in missing[: int(max(1, cfg.debug_max_names_per_group))]]
                            suffix = " ..." if len(missing) > int(max(1, cfg.debug_max_names_per_group)) else ""
                            print(f"  {g}: missing_abs_indices={missing[:10]} names={names}{suffix}")
                        else:
                            print(f"  {g}: missing_abs_indices={missing[:10]}")

                # Compute per-entry deltas inside the published upper_body vector.
                upper_before = q[upper_idx].astype(np.float32, copy=False)
                upper_delta = target_upper - upper_before

                # Per-group stats (in joint-space of each group, not necessarily contiguous).
                def _group_delta_stats(group_name: str) -> Dict[str, float]:
                    g_abs = idx[group_name]
                    # Extract the entries of upper_delta corresponding to this group.
                    d = []
                    for j in g_abs.tolist():
                        j = int(j)
                        if j in pos_map:
                            d.append(float(abs(upper_delta[pos_map[j]])))
                    if not d:
                        return {"max": 0.0, "mean": 0.0}
                    return {"max": float(max(d)), "mean": float(sum(d) / len(d))}

                if tick % int(max(1, cfg.debug_print_every)) == 0:
                    print("--- mapping debug ---")
                    print("apply_groups:", apply_groups)
                    for g in apply_groups:
                        stats = _group_delta_stats(g)
                        print(f"  {g}: |Δ| max={stats['max']:.6f} mean={stats['mean']:.6f}")

                # Detect any changed joints in upper_body that are NOT part of the intended groups.
                changed_unexpected = []
                for local_i, abs_joint_i in enumerate(upper_idx.tolist()):
                    abs_joint_i = int(abs_joint_i)
                    if abs(float(upper_delta[local_i])) > float(cfg.debug_eps):
                        if abs_joint_i not in intended_abs:
                            changed_unexpected.append(
                                (abs_joint_i, float(upper_delta[local_i]), _name_for_index(abs_joint_i))
                            )

                if changed_unexpected and tick % int(max(1, cfg.debug_print_every)) == 0:
                    # If this triggers, something is wrong in the mapping logic, or upper_body contains
                    # extra joints that are being modified unexpectedly.
                    print(
                        "WARNING: unexpected changed joints inside upper_body (abs_index, delta, name):"
                    )
                    for abs_i, d, nm in changed_unexpected[:20]:
                        print(f"  {abs_i}: Δ={d:.6f} ({nm})")
                    if len(changed_unexpected) > 20:
                        print(f"  ... and {len(changed_unexpected) - 20} more")

                    if cfg.debug_fail_on_unexpected_joint_change:
                        raise RuntimeError(
                            "Unexpected joint changes detected inside upper_body. "
                            "This indicates a mapping bug; see warnings above."
                        )

            # If you're debugging, you can compute everything but skip publishing.
            if cfg.dry_run_print_only:
                print("Would publish target_upper (first 5):", target_upper[:5])
                rate.sleep()
                continue

            # Optional locomotion channels (use defaults if absent)
            # NOTE: This bridge primarily drives the upper body. The lower body is still
            # managed by the WBC control loop's lower-body policy.
            base_h_key = None
            for k in (
                "base_height_command",
                "action.base_height_command",
            ):
                if k in action_dict:
                    base_h_key = k
                    break
            if base_h_key is not None:
                base_h_arr = _as_btd(action_dict[base_h_key])
                t_clamped = int(np.clip(t, 0, base_h_arr.shape[1] - 1))
                base_h = float(base_h_arr[0, t_clamped, 0])
            else:
                base_h = float(DEFAULT_BASE_HEIGHT)

            nav_key = None
            for k in (
                "navigate_cmd",
                "navigate_command",
                "action.navigate_cmd",
                "action.navigate_command",
            ):
                if k in action_dict:
                    nav_key = k
                    break
            if nav_key is not None:
                nav_arr = _as_btd(action_dict[nav_key])
                t_clamped = int(np.clip(t, 0, nav_arr.shape[1] - 1))
                nav = nav_arr[0, t_clamped, :].astype(float).tolist()
            else:
                nav = list(DEFAULT_NAV_CMD)

            msg = {
                # Upper-body goal: joint targets in the `upper_body` joint-group order.
                "target_upper_body_pose": target_upper,
                "wrist_pose": DEFAULT_WRIST_POSE,
                "base_height_command": base_h,
                "navigate_cmd": nav,
                # Required for InterpolationPolicy (default upper-body policy).
                # Must be monotonic time to avoid wall-clock adjustments breaking interpolation.
                "target_time": time.monotonic() + float(cfg.target_time_offset_s),
                # Wall clock timestamp is optional; it's useful for logging only.
                "timestamp": time.time(),
            }

            if tick == 0:
               print("Publishing keys:", list(msg.keys()))
               print("target_upper_body_pose len:", len(msg["target_upper_body_pose"]))
               print("target_upper_body_pose first5:", np.asarray(msg["target_upper_body_pose"])[:5])

            goal_pub.publish(msg)

            tick += 1
            if tick % int(max(1, cfg.rate_hz * 2)) == 0:  # ~every 2 seconds
                print(
                    f"tick={tick} published upper_body[{target_upper.shape[0]}], "
                    f"base_h={base_h:.3f}, nav={nav}"
                )

            rate.sleep()

    finally:
        ros_manager.shutdown()


if __name__ == "__main__":
    cfg = tyro.cli(Cfg)
    main(cfg)
