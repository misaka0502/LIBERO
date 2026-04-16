import argparse
import atexit
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np
import robosuite.macros as macros
import robosuite.utils.transform_utils as T
import torch

import init_path
from robosuite import load_controller_config

from libero.libero.envs import TASK_MAPPING
import libero.libero.envs.bddl_utils as BDDLUtils

from create_pcd_dataset import (
    build_gripper_spec,
    build_latent_pose_alignment,
    compute_pairwise_distance_traj,
    gripper_pose_quat_offset_wxyz,
    normalize_quaternion_wxyz,
    quat_mul_wxyz,
    transform_point_cloud,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
WM_ROOT = REPO_ROOT / "WM"
if str(WM_ROOT) not in sys.path:
    sys.path.append(str(WM_ROOT))
DEFAULT_ACTION_DATASET_DIRS = [
    Path("/home/franka-client/datasets/libero/libero_object"),
    Path("/home/franka-client/datasets/libero/libero_spatial"),
    Path("/home/franka-client/datasets/libero/libero_10"),
]


def build_empty_kitchen_bddl_text() -> str:
    """Return a minimal kitchen-table BDDL with no movable objects."""
    return textwrap.dedent(
        """
        (define (problem LIBERO_Kitchen_Tabletop_Manipulation)
          (:domain robosuite)
          (:language Move the gripper over an empty kitchen table)

          (:regions
            (workspace_region
                (:target kitchen_table)
                (:ranges (
                    (-0.20 -0.20 0.20 0.20)
                  )
                )
            )
          )

          (:fixtures
            kitchen_table - kitchen_table
          )

          (:objects
          )

          (:obj_of_interest
          )

          (:init
          )

          (:goal
            (And)
          )
        )
        """
    ).strip() + "\n"


def write_empty_kitchen_bddl(output_path: Optional[str] = None) -> str:
    """Write the minimal BDDL to disk and return the path."""
    if output_path is not None:
        out_path = Path(output_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(build_empty_kitchen_bddl_text(), encoding="utf-8")
        return str(out_path)

    fd, tmp_path = tempfile.mkstemp(prefix="gripper_only_kitchen_", suffix=".bddl")
    os.close(fd)
    Path(tmp_path).write_text(build_empty_kitchen_bddl_text(), encoding="utf-8")
    atexit.register(lambda path=tmp_path: os.path.exists(path) and os.remove(path))
    return tmp_path


def build_env(
    bddl_file_name: str,
    controller: str = "OSC_POSE",
    use_camera_obs: bool = False,
    camera_name: str = "agentview",
):
    """
    Create a gripper-only LIBERO kitchen tabletop environment.

    This reuses the standard LIBERO kitchen scene, so the table placement,
    robot mount, and workspace offset stay aligned with the original tasks.
    """
    controller_config = load_controller_config(default_controller=controller)
    problem_name = "libero_kitchen_tabletop_manipulation"
    env = TASK_MAPPING[problem_name](
        bddl_file_name=bddl_file_name,
        robots=["Panda"],
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=use_camera_obs,
        render_camera=camera_name,
        ignore_done=True,
        reward_shaping=False,
        control_freq=20,
        camera_names=[camera_name],
        camera_heights=256,
        camera_widths=256,
    )
    return env


def safe_reset(env, max_tries: int = 10):
    last_error = None
    for _ in range(max_tries):
        try:
            return env.reset()
        except Exception as exc:  # pragma: no cover - defensive reset retry
            last_error = exc
    raise RuntimeError(f"Failed to reset environment after {max_tries} tries") from last_error


def extract_env_summary(env, obs) -> dict:
    summary = {
        "action_dim": int(env.action_dim),
        "ee_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float64).tolist()
        if "robot0_eef_pos" in obs
        else None,
        "ee_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float64).tolist()
        if "robot0_eef_quat" in obs
        else None,
        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).tolist()
        if "robot0_gripper_qpos" in obs
        else None,
        "workspace_name": getattr(env, "workspace_name", None),
        "workspace_offset": np.asarray(getattr(env, "workspace_offset", [0, 0, 0]), dtype=np.float64).tolist(),
        "scene_xml": getattr(env, "_arena_xml", None),
    }
    return summary


class CurrentEnvPoseExtractor:
    def __init__(self, env):
        self.env = env
        self.model = env.sim.model
        self.data = env.sim.data
        self.body_name_to_idx = {}
        self.joint_name_to_idx = {}
        self.gripper_joint_names = []
        self._init_body_and_joint_maps()

    def _init_body_and_joint_maps(self):
        model = self.model
        for i in range(int(model.nbody)):
            adr = int(model.name_bodyadr[i])
            if adr <= 0:
                continue
            name = model.names[adr:].split(b"\x00")[0].decode("utf-8")
            self.body_name_to_idx[name] = int(i)

        finger_joint_candidates = []
        for i in range(int(model.njnt)):
            adr = int(model.name_jntadr[i])
            if adr <= 0:
                continue
            name = model.names[adr:].split(b"\x00")[0].decode("utf-8")
            self.joint_name_to_idx[name] = int(i)
            lname = name.lower()
            if "gripper" in lname and "finger_joint" in lname:
                finger_joint_candidates.append((int(i), name))
        if len(finger_joint_candidates) < 2:
            for jname, jidx in self.joint_name_to_idx.items():
                lname = jname.lower()
                if "finger" in lname and "joint" in lname:
                    finger_joint_candidates.append((int(jidx), jname))
        finger_joint_candidates.sort(key=lambda x: x[0])
        self.gripper_joint_names = [name for _, name in finger_joint_candidates[:2]]

    def extract_current_poses(self):
        poses = {}
        for body_name, body_idx in self.body_name_to_idx.items():
            pos = np.asarray(self.env.sim.data.body_xpos[body_idx], dtype=np.float32).copy()
            quat = np.asarray(self.env.sim.data.body_xquat[body_idx], dtype=np.float32).copy()
            poses[body_name] = (pos, quat)
        return poses

    def get_gripper_width(self):
        if len(self.gripper_joint_names) < 2:
            return None
        vals = []
        for joint_name in self.gripper_joint_names[:2]:
            try:
                vals.append(float(self.env.sim.data.get_joint_qpos(joint_name)))
            except Exception:
                continue
        if len(vals) < 2:
            return None
        return float(abs(vals[0] - vals[1]))


def print_env_summary(summary: dict, obs, print_obs_keys: bool) -> None:
    print("[INFO] Empty kitchen-table gripper-only environment ready.")
    print(json.dumps(summary, indent=2))
    if print_obs_keys:
        print("[INFO] Observation keys:")
        for key in sorted(obs.keys()):
            print(f"  - {key}")


def build_gripper_spec_from_env(env, num_points_per_entity: int):
    pose_extractor = CurrentEnvPoseExtractor(env)
    spec = build_gripper_spec(pose_extractor, int(num_points_per_entity))
    if spec is None:
        raise RuntimeError("Failed to build gripper specification from the current environment.")
    return spec, pose_extractor


def collect_episode_data(
    env,
    actions: np.ndarray,
    gripper_spec: dict,
    pose_extractor: CurrentEnvPoseExtractor,
    latent_pose_alignment: Optional[dict],
    show: bool,
    camera_name: str,
    sim_image_width: int,
    sim_image_height: int,
    window_name: str,
) -> dict:
    gripper_entities = list(gripper_spec["entities"])
    gripper_root_name = str(gripper_spec["root_body_name"])
    num_steps = int(actions.shape[0])
    num_points = int(np.asarray(gripper_entities[0]["points"]).shape[0])

    object_pcds = {
        str(gspec["entity_name"]): np.zeros((num_steps, num_points, 3), dtype=np.float32)
        for gspec in gripper_entities
    }
    object_poses = {str(gspec["entity_name"]): [] for gspec in gripper_entities}
    object_poses_aligned = {str(gspec["entity_name"]): [] for gspec in gripper_entities}
    gripper_pose_list = []
    gripper_width_traj = np.zeros((num_steps, 1), dtype=np.float32)

    states = []
    ee_states = []
    joint_states = []
    gripper_states = []
    robot_states = []
    rewards = []
    dones = np.zeros((num_steps,), dtype=np.uint8)
    init_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
    model_xml = env.sim.model.get_xml()

    for t in range(num_steps):
        poses = pose_extractor.extract_current_poses()
        state_flat = np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
        states.append(state_flat)
        sim_width = pose_extractor.get_gripper_width()
        if sim_width is None:
            sim_width = 0.0
        gripper_width_traj[t, 0] = float(sim_width)

        for gspec in gripper_entities:
            entity_name = str(gspec["entity_name"])
            body_name = str(gspec["body_name"])
            pos, quat = poses.get(
                body_name,
                (
                    np.array([0.0, 0.0, 0.0], dtype=np.float32),
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                ),
            )
            quat_raw = normalize_quaternion_wxyz(quat).astype(np.float32)
            q_off = gripper_pose_quat_offset_wxyz(gspec.get("part", ""), body_name).astype(np.float32)
            quat_aligned = quat_mul_wxyz(quat_raw, q_off).astype(np.float32)
            center_local = np.asarray(gspec.get("surface_center", np.zeros((3,), dtype=np.float32)), dtype=np.float32)
            pose_pos = transform_point_cloud(center_local.reshape(1, 3), pos, quat_raw)[0].astype(np.float32, copy=False)
            align_info = latent_pose_alignment.get(entity_name, None) if latent_pose_alignment else None
            if align_info is not None:
                align_t = np.asarray(
                    align_info.get("trans_local", np.zeros((3,), dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(3)
                align_q = np.asarray(
                    align_info.get("rot_wxyz", np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(4)
            else:
                align_t = center_local.reshape(3)
                align_q = q_off
            pose_pos_aligned = transform_point_cloud(align_t.reshape(1, 3), pos, quat_raw)[0].astype(np.float32, copy=False)
            pose_quat_aligned = quat_mul_wxyz(quat_raw, align_q).astype(np.float32)
            raw_points = np.asarray(gspec.get("raw_points", gspec["points"]), dtype=np.float32)
            transformed = transform_point_cloud(raw_points, pos, quat_raw).astype(np.float32)

            object_pcds[entity_name][t] = transformed
            object_poses[entity_name].append(np.concatenate([pose_pos, quat_aligned], axis=0).astype(np.float32))
            object_poses_aligned[entity_name].append(
                np.concatenate([pose_pos_aligned, pose_quat_aligned], axis=0).astype(np.float32)
            )

        root_pos, root_quat = poses.get(
            gripper_root_name,
            (
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
        )
        gripper_pose_list.append(
            np.concatenate([np.asarray(root_pos, dtype=np.float32), np.asarray(root_quat, dtype=np.float32)], axis=0)
        )

        action = np.asarray(actions[t], dtype=np.float32)
        obs, reward, _done, _info = env.step(action)

        if show:
            frame_bgr = render_sim_image(
                env,
                camera_name=camera_name,
                width=sim_image_width,
                height=sim_image_height,
            )
            cv2.imshow(window_name, frame_bgr)
            cv2.waitKey(1)

        rewards.append(float(reward))

        if "robot0_gripper_qpos" in obs:
            gripper_states.append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).copy())
        if "robot0_joint_pos" in obs:
            joint_states.append(np.asarray(obs["robot0_joint_pos"], dtype=np.float32).copy())
        if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
            ee_states.append(
                np.hstack(
                    (
                        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                        T.quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
                    )
                ).astype(np.float32)
            )
        try:
            robot_states.append(np.asarray(env.get_robot_state_vector(obs), dtype=np.float32).copy())
        except Exception:
            pass

    if num_steps > 0:
        dones[-1] = 1
    distance_entity_names = sorted(object_pcds.keys())
    dist_values = compute_pairwise_distance_traj(object_pcds, distance_entity_names)

    return {
        "init_state": init_state,
        "model_xml": model_xml,
        "actions": np.asarray(actions, dtype=np.float64),
        "states": np.asarray(states, dtype=np.float64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": dones,
        "robot_states": np.asarray(robot_states, dtype=np.float32) if robot_states else None,
        "obs_gripper_states": np.asarray(gripper_states, dtype=np.float32) if gripper_states else None,
        "obs_joint_states": np.asarray(joint_states, dtype=np.float32) if joint_states else None,
        "obs_ee_states": np.asarray(ee_states, dtype=np.float32) if ee_states else None,
        "object_pcds": object_pcds,
        "object_poses": {k: np.stack(v, axis=0).astype(np.float32) for k, v in object_poses.items()},
        "object_poses_aligned": {k: np.stack(v, axis=0).astype(np.float32) for k, v in object_poses_aligned.items()},
        "gripper_poses": np.stack(gripper_pose_list, axis=0).astype(np.float32),
        "gripper_width": gripper_width_traj.astype(np.float32),
        "distance_entity_names": distance_entity_names,
        "distance_values": np.asarray(dist_values, dtype=np.float32),
    }


def build_gripper_latent_mapping(gripper_entities, latent_ckpt_path: Optional[str]) -> Optional[dict]:
    if not latent_ckpt_path:
        return None

    from train import (
        LatentMatcher,
        build_latent_entries,
        match_entity_to_object_id,
        parse_entity_name,
    )

    ckpt = torch.load(latent_ckpt_path, map_location="cpu", weights_only=False)
    object_meta = ckpt.get("object_meta")
    if not isinstance(object_meta, list):
        raise RuntimeError(f"{latent_ckpt_path} does not contain object_meta.")

    entries = build_latent_entries(object_meta)
    matcher = LatentMatcher(entries, min_score=8.0)
    object_meta_by_id = {
        int(item.get("object_id", i)): item
        for i, item in enumerate(object_meta)
        if isinstance(item, dict)
    }

    rows = []
    latent_mapping_by_name = {}
    for gspec in gripper_entities:
        entity_name = str(gspec["entity_name"])
        inst_name, part_name = parse_entity_name(entity_name)
        oid, score = match_entity_to_object_id(
            matcher=matcher,
            entity_name=entity_name,
            instance_name=inst_name,
            category_name="gripper",
            part_name=part_name,
        )
        if oid is None:
            raise RuntimeError(f"Could not match gripper entity {entity_name} to latent object_id from {latent_ckpt_path}.")
        meta = object_meta_by_id.get(int(oid), {})
        npz_path = str(meta.get("npz_path", ""))
        target_npz_name = os.path.basename(npz_path.replace("\\", "/")) if npz_path else ""
        q_off = gripper_pose_quat_offset_wxyz(gspec.get("part", ""), str(gspec["body_name"])).astype(np.float32)
        row = {
            "entity_name": entity_name,
            "pose_source_name": entity_name,
            "target_npz_name": target_npz_name,
            "target_object_id": int(oid),
            "pose_offset_local": np.asarray(gspec.get("surface_center", np.zeros((3,), dtype=np.float32)), dtype=np.float32),
            "pose_rot_offset_wxyz": q_off,
            "alignment_mode": "gripper-surface-center",
            "body_name": str(gspec["body_name"]),
            "instance": inst_name,
            "category": "gripper",
            "part": str(gspec.get("part", "")),
            "match_kind": "direct_gripper",
            "pose_align_error": float(max(0.0, 8.0 - float(score))),
        }
        rows.append(row)
        latent_mapping_by_name[entity_name] = {
            "object_id": int(oid),
            "target_npz_name": target_npz_name,
            "pose_source": entity_name,
            "pose_offset_local": np.asarray(row["pose_offset_local"], dtype=np.float32).reshape(3),
            "pose_rot_offset_wxyz": np.asarray(row["pose_rot_offset_wxyz"], dtype=np.float32).reshape(4),
            "body_name": row["body_name"],
            "instance": row["instance"],
            "category": row["category"],
            "part": row["part"],
            "match_kind": row["match_kind"],
            "pose_align_error": row["pose_align_error"],
            "alignment_mode": row["alignment_mode"],
        }

    align_info = build_latent_pose_alignment(
        entity_specs={},
        gripper_spec={"entities": list(gripper_entities)},
        latent_mapping=latent_mapping_by_name,
        object_meta=object_meta,
        num_points_per_entity=int(np.asarray(gripper_entities[0]["points"]).shape[0]) if gripper_entities else 1024,
    )
    for row in rows:
        entity_name = row["entity_name"]
        info = align_info.get(entity_name, None)
        if info is None:
            continue
        row["pose_offset_local"] = np.asarray(
            info.get("trans_local", row["pose_offset_local"]),
            dtype=np.float32,
        ).reshape(3)
        row["pose_rot_offset_wxyz"] = np.asarray(
            info.get("rot_wxyz", row["pose_rot_offset_wxyz"]),
            dtype=np.float32,
        ).reshape(4)
        row["pose_align_error"] = float(info.get("fit_error", row["pose_align_error"]))
        row["alignment_mode"] = str(info.get("alignment_mode", row["alignment_mode"]))
    return {"rows": rows}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _make_env_args(bddl_path: str, env) -> dict:
    problem_info = BDDLUtils.get_problem_info(bddl_path)
    env_name = env.__class__.__name__
    robot0 = getattr(env, "robots", [None])[0]
    controller_cfg = getattr(robot0, "controller_config", None)
    controller_name = None
    if isinstance(controller_cfg, dict):
        controller_name = str(controller_cfg.get("type", "") or controller_cfg.get("name", ""))
    if not controller_name:
        controller_name = str(getattr(getattr(robot0, "controller", None), "name", "") or "")
    return {
        "type": 1,
        "env_name": env_name,
        "problem_name": problem_info["problem_name"],
        "bddl_file": str(bddl_path),
        "env_kwargs": {
            "bddl_file_name": str(bddl_path),
            "robots": ["Panda"],
            "controller": controller_name,
        },
    }

def initialize_gripper_only_hdf5(
    output_path: str,
    bddl_path: str,
    env,
    latent_mapping: Optional[dict],
) -> str:
    output_path = str(Path(output_path).expanduser().resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    env_args = _make_env_args(bddl_path, env)
    problem_info = BDDLUtils.get_problem_info(bddl_path)
    env_name = env.__class__.__name__
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as f:
        data_group = f.create_group("data")
        data_group.attrs["env_name"] = env_name
        data_group.attrs["problem_info"] = json.dumps(problem_info)
        data_group.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION
        data_group.attrs["env_args"] = json.dumps(_jsonable(env_args))
        data_group.attrs["bddl_file_name"] = str(bddl_path)
        data_group.attrs["num_demos"] = 0
        data_group.attrs["total"] = 0
        data_group.attrs["tag"] = "gripper_only"

        if latent_mapping is not None:
            meta_group = f.create_group("meta")
            mapping_group = meta_group.create_group("latent_mapping")
            rows = list(latent_mapping["rows"])
            mapping_group.create_dataset("entity_names", data=np.asarray([r["entity_name"] for r in rows], dtype=str_dtype))
            mapping_group.create_dataset("pose_source_names", data=np.asarray([r["pose_source_name"] for r in rows], dtype=str_dtype))
            mapping_group.create_dataset("target_npz_names", data=np.asarray([r["target_npz_name"] for r in rows], dtype=str_dtype))
            mapping_group.create_dataset("target_object_ids", data=np.asarray([r["target_object_id"] for r in rows], dtype=np.int32))
            mapping_group.create_dataset("pose_offset_local", data=np.stack([r["pose_offset_local"] for r in rows], axis=0).astype(np.float32))
            mapping_group.create_dataset("pose_rot_offset_wxyz", data=np.stack([r["pose_rot_offset_wxyz"] for r in rows], axis=0).astype(np.float32))
            mapping_group.create_dataset("alignment_mode", data=np.asarray([r["alignment_mode"] for r in rows], dtype=str_dtype))
            mapping_group.create_dataset("body_names", data=np.asarray([r["body_name"] for r in rows], dtype=str_dtype))
            mapping_group.create_dataset("instances", data=np.asarray([r["instance"] for r in rows], dtype=str_dtype))
            mapping_group.create_dataset("categories", data=np.asarray([r["category"] for r in rows], dtype=str_dtype))
            mapping_group.create_dataset("parts", data=np.asarray([r["part"] for r in rows], dtype=str_dtype))
            mapping_group.create_dataset("match_kind", data=np.asarray([r["match_kind"] for r in rows], dtype=str_dtype))
            mapping_group.create_dataset("pose_align_error", data=np.asarray([r["pose_align_error"] for r in rows], dtype=np.float32))
            mapping_group.attrs["latent_ckpt"] = str(latent_mapping.get("latent_ckpt", ""))
            mapping_group.attrs["mapping_version"] = 4
    return output_path


def append_gripper_only_episode(output_path: str, ep_idx: int, ep: dict) -> None:
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "a") as f:
        data_group = f["data"]
        demo_group = data_group.create_group(f"demo_{ep_idx}")
        demo_group.attrs["init_state"] = np.asarray(ep["init_state"], dtype=np.float64)
        demo_group.attrs["model_file"] = str(ep["model_xml"])
        demo_group.attrs["num_samples"] = int(ep["actions"].shape[0])

        demo_group.create_dataset("actions", data=np.asarray(ep["actions"], dtype=np.float64), compression="gzip")
        demo_group.create_dataset("states", data=np.asarray(ep["states"], dtype=np.float64), compression="gzip")
        demo_group.create_dataset("rewards", data=np.asarray(ep["rewards"], dtype=np.float32), compression="gzip")
        demo_group.create_dataset("dones", data=np.asarray(ep["dones"], dtype=np.uint8), compression="gzip")
        if ep["robot_states"] is not None:
            demo_group.create_dataset("robot_states", data=np.asarray(ep["robot_states"], dtype=np.float32), compression="gzip")

        obs_group = demo_group.create_group("obs")
        if ep["obs_gripper_states"] is not None:
            obs_group.create_dataset("gripper_states", data=np.asarray(ep["obs_gripper_states"], dtype=np.float32), compression="gzip")
        if ep["obs_joint_states"] is not None:
            obs_group.create_dataset("joint_states", data=np.asarray(ep["obs_joint_states"], dtype=np.float32), compression="gzip")
        if ep["obs_ee_states"] is not None:
            ee = np.asarray(ep["obs_ee_states"], dtype=np.float32)
            obs_group.create_dataset("ee_states", data=ee, compression="gzip")
            obs_group.create_dataset("ee_pos", data=ee[:, :3], compression="gzip")
            obs_group.create_dataset("ee_ori", data=ee[:, 3:], compression="gzip")

        pcd_group = demo_group.create_group("object_pcds")
        for entity_name, arr in ep["object_pcds"].items():
            pcd_group.create_dataset(entity_name, data=np.asarray(arr, dtype=np.float32), compression="gzip")

        pose_group = demo_group.create_group("object_poses")
        for entity_name, arr in ep["object_poses"].items():
            pose_group.create_dataset(entity_name, data=np.asarray(arr, dtype=np.float32), compression="gzip")

        pose_aligned_group = demo_group.create_group("object_poses_aligned")
        for entity_name, arr in ep["object_poses_aligned"].items():
            pose_aligned_group.create_dataset(entity_name, data=np.asarray(arr, dtype=np.float32), compression="gzip")

        gripper_pose_group = demo_group.create_group("gripper_poses")
        gripper_pose_group.create_dataset("robot_gripper", data=np.asarray(ep["gripper_poses"], dtype=np.float32), compression="gzip")
        demo_group.create_dataset("gripper_width", data=np.asarray(ep["gripper_width"], dtype=np.float32), compression="gzip")

        dist_group = demo_group.create_group("object_distances")
        dist_group.create_dataset("entity_names", data=np.asarray(ep["distance_entity_names"], dtype=str_dtype))
        dist_group.create_dataset("values", data=np.asarray(ep["distance_values"], dtype=np.float32), compression="gzip")

        data_group.attrs["num_demos"] = int(data_group.attrs.get("num_demos", 0)) + 1
        data_group.attrs["total"] = int(data_group.attrs.get("total", 0)) + int(ep["actions"].shape[0])


def render_sim_image(env, camera_name: str, width: int, height: int) -> np.ndarray:
    frame_rgb = env.sim.render(
        camera_name=camera_name,
        width=max(1, int(width)),
        height=max(1, int(height)),
    )
    frame_bgr = frame_rgb[..., ::-1]
    return np.ascontiguousarray(np.flip(frame_bgr, axis=0))


def iter_demo_hdf5_files(dataset_dirs):
    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir).expanduser()
        if not dataset_dir.exists():
            continue
        for path in sorted(dataset_dir.glob("*.hdf5")):
            yield path


def load_action_clips(dataset_dirs, min_length: int = 32):
    clips = []
    source_files = []
    for path in iter_demo_hdf5_files(dataset_dirs):
        try:
            with h5py.File(path, "r") as f:
                if "data" not in f:
                    continue
                for ep in f["data"].keys():
                    dset = f[f"data/{ep}/actions"]
                    actions = np.asarray(dset[()], dtype=np.float32)
                    if actions.ndim != 2 or actions.shape[1] < 7 or actions.shape[0] < min_length:
                        continue
                    clips.append(actions[:, :7].copy())
                    source_files.append(f"{path.name}:{ep}")
        except Exception:
            continue
    return clips, source_files


def compute_action_clip_stats(action_clips):
    if not action_clips:
        raise RuntimeError("Cannot compute action statistics from an empty clip list.")
    all_actions = np.concatenate(action_clips, axis=0).astype(np.float32, copy=False)
    return {
        "p90_abs": np.quantile(np.abs(all_actions[:, :6]), 0.90, axis=0).astype(np.float32),
        "p95_abs": np.quantile(np.abs(all_actions[:, :6]), 0.95, axis=0).astype(np.float32),
        "p99_abs": np.quantile(np.abs(all_actions[:, :6]), 0.99, axis=0).astype(np.float32),
        "abs_max": np.max(np.abs(all_actions[:, :6]), axis=0).astype(np.float32),
    }


def sample_realistic_action_sequence(
    steps: int,
    action_dim: int,
    rng: np.random.Generator,
    action_clips,
    action_stats,
    gripper_mode: str,
    real_action_scale: float,
    speed_mix=(0.20, 0.60, 0.20),
) -> np.ndarray:
    """
    Build a realistic action sequence by stitching together real action clips
    sampled from the original LIBERO datasets.
    """
    if action_dim < 7:
        raise ValueError(f"Expected at least 7 action dims, got {action_dim}")
    if not action_clips:
        raise RuntimeError("No real action clips available for realistic action sampling.")

    steps = max(1, int(steps))
    seq = np.zeros((steps, action_dim), dtype=np.float32)
    cursor = 0

    while cursor < steps:
        clip = action_clips[int(rng.integers(0, len(action_clips)))]
        # Favor medium-length windows so the motion has intent instead of looking jittery.
        max_seg_len = min(int(clip.shape[0]), max(24, steps - cursor))
        min_seg_len = min(max(12, steps // 8), max_seg_len)
        if min_seg_len <= 0:
            break
        seg_len = int(rng.integers(min_seg_len, max_seg_len + 1))
        start_hi = max(1, int(clip.shape[0]) - seg_len + 1)
        start = int(rng.integers(0, start_hi))
        seg = np.asarray(clip[start : start + seg_len], dtype=np.float32).copy()

        # Explicitly mix slow / medium / fast motion segments so the dataset
        # contains a broader range of gripper speeds while still staying within
        # a conservative envelope derived from real demonstrations.
        speed_mode = str(
            rng.choice(
                ["slow", "medium", "fast"],
                p=list(speed_mix),
            )
        )
        if speed_mode == "slow":
            clip_limit = np.asarray(action_stats["p90_abs"], dtype=np.float32)
            scale_lo, scale_hi = 0.35, 0.60
        elif speed_mode == "fast":
            clip_limit = np.asarray(action_stats["p99_abs"], dtype=np.float32)
            scale_lo, scale_hi = 0.90, 1.15
        else:
            clip_limit = np.asarray(action_stats["p95_abs"], dtype=np.float32)
            scale_lo, scale_hi = 0.60, 0.90

        # Mild amplitude perturbation keeps diversity without losing realism.
        pos_scale = float(real_action_scale) * float(rng.uniform(scale_lo, scale_hi))
        rot_scale = float(real_action_scale) * float(rng.uniform(scale_lo, scale_hi))
        seg[:, :3] *= pos_scale
        seg[:, 3:6] *= rot_scale

        # A light temporal smoothing keeps clip stitching from producing visible kinks.
        for t in range(1, seg.shape[0]):
            seg[t, :6] = 0.85 * seg[t - 1, :6] + 0.15 * seg[t, :6]

        # Keep commands within a conservative envelope derived from real data.
        seg[:, :6] = np.clip(seg[:, :6], -clip_limit[None, :], clip_limit[None, :])

        copy_len = min(seg.shape[0], steps - cursor)
        seq[cursor : cursor + copy_len, :7] = seg[:copy_len, :7]
        cursor += copy_len

    if gripper_mode == "open":
        seq[:, 6] = -1.0
    elif gripper_mode == "close":
        seq[:, 6] = 1.0
    else:
        # keep gripper command from real clips
        seq[:, 6] = np.where(seq[:, 6] >= 0.0, 1.0, -1.0)

    return seq


def sample_mixed_action_sequence(
    steps: int,
    action_dim: int,
    rng: np.random.Generator,
    action_clips,
    action_stats,
    gripper_mode: str,
    real_action_scale: float,
) -> Tuple[np.ndarray, str]:
    """
    Mix three distributions:
    - real: conservative clip-based motion close to the dataset
    - real_wide: clip-based motion with wider amplitude / faster segments
    - explore: structured workspace exploration
    """
    mode = str(
        rng.choice(
            ["real", "real_wide", "explore"],
            p=[0.45, 0.25, 0.30],
        )
    )
    if mode == "real":
        actions = sample_realistic_action_sequence(
            steps=steps,
            action_dim=action_dim,
            rng=rng,
            action_clips=action_clips,
            action_stats=action_stats,
            gripper_mode=gripper_mode,
            real_action_scale=real_action_scale,
            speed_mix=(0.25, 0.60, 0.15),
        )
    elif mode == "real_wide":
        actions = sample_realistic_action_sequence(
            steps=steps,
            action_dim=action_dim,
            rng=rng,
            action_clips=action_clips,
            action_stats=action_stats,
            gripper_mode=gripper_mode,
            real_action_scale=min(1.0, float(real_action_scale) * 1.15),
            speed_mix=(0.15, 0.45, 0.40),
        )
    else:
        explore_limits = 0.85 * np.asarray(action_stats["p95_abs"], dtype=np.float32)
        actions = sample_smooth_action_sequence(
            steps=steps,
            action_dim=action_dim,
            rng=rng,
            pos_limit=float(np.max(explore_limits[:3])),
            rot_limit=float(np.max(explore_limits[3:6])),
            min_hold=12,
            max_hold=28,
            smoothing_alpha=0.68,
            gripper_mode=gripper_mode,
            clip_limits_6=explore_limits,
            target_scale=0.55,
        )
    return actions, mode


def sample_smooth_action_sequence(
    steps: int,
    action_dim: int,
    rng: np.random.Generator,
    pos_limit: float,
    rot_limit: float,
    min_hold: int,
    max_hold: int,
    smoothing_alpha: float,
    gripper_mode: str,
    clip_limits_6: Optional[np.ndarray] = None,
    target_scale: float = 1.0,
) -> np.ndarray:
    """
    Generate a smooth, random, bounded action sequence.

    Layout follows LIBERO's 7D action semantics:
    - [:3] Cartesian delta
    - [3:6] rotation delta
    - [6] binary gripper command in {-1, +1}
    """
    if action_dim < 7:
        raise ValueError(f"Expected at least 7 action dims, got {action_dim}")

    steps = max(1, int(steps))
    seq = np.zeros((steps, action_dim), dtype=np.float32)

    def _sample_directional_target(limit: float, dims: int) -> np.ndarray:
        # Prefer axis-aligned and diagonal commands over isotropic jitter so the
        # arm clearly travels through the workspace instead of hovering in place.
        mode = float(rng.random())
        if dims == 3 and mode < 0.45:
            axis = int(rng.integers(0, 3))
            vec = np.zeros((3,), dtype=np.float32)
            vec[axis] = -1.0 if float(rng.random()) < 0.5 else 1.0
        elif dims == 3 and mode < 0.80:
            axes = rng.choice(3, size=2, replace=False)
            vec = np.zeros((3,), dtype=np.float32)
            for axis in axes:
                vec[int(axis)] = -1.0 if float(rng.random()) < 0.5 else 1.0
            vec /= max(np.linalg.norm(vec), 1e-6)
        else:
            vec = rng.normal(size=(dims,)).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm < 1e-8:
                vec = np.zeros((dims,), dtype=np.float32)
            else:
                vec /= norm
        mag = float(rng.uniform(0.35, 1.0)) * float(limit) * float(target_scale)
        if float(rng.random()) < 0.12:
            mag *= 0.2
        return np.asarray(vec * mag, dtype=np.float32)

    def _fill_motion_primitives(dim_slice, limit: float, dof_name: str):
        t = 0
        current = np.zeros((dim_slice.stop - dim_slice.start,), dtype=np.float32)
        while t < steps:
            hold = int(rng.integers(min_hold, max_hold + 1))
            # Translation should stay on one primitive longer so the arm
            # actually sweeps across the scene.
            if dof_name == "pos":
                hold += int(rng.integers(min_hold, max_hold + 1))
            next_t = min(steps, t + hold)
            target = _sample_directional_target(limit=limit, dims=current.shape[0])

            # Ramp into the primitive quickly, then keep a stable command.
            ramp = max(2, min(6, next_t - t))
            span = max(1, next_t - t)
            for k in range(span):
                if k < ramp:
                    frac = float(k + 1) / float(ramp)
                    seq[t + k, dim_slice] = (1.0 - frac) * current + frac * target
                else:
                    seq[t + k, dim_slice] = target
            current = target
            t = next_t

    _fill_motion_primitives(slice(0, 3), pos_limit, "pos")
    _fill_motion_primitives(slice(3, 6), rot_limit, "rot")

    # Low-pass filter to remove sharp changes between segments.
    alpha = float(np.clip(smoothing_alpha, 0.0, 0.999))
    if alpha > 0.0:
        for t in range(1, steps):
            seq[t, :6] = alpha * seq[t - 1, :6] + (1.0 - alpha) * seq[t, :6]

    # Keep positional commands from collapsing into tiny values after filtering.
    pos_norm = np.linalg.norm(seq[:, :3], axis=1)
    low_motion = pos_norm < (0.25 * float(pos_limit))
    if np.any(low_motion):
        seq[low_motion, :3] *= 1.5
        seq[:, :3] = np.clip(seq[:, :3], -float(pos_limit), float(pos_limit))

    if clip_limits_6 is not None:
        clip_limits_6 = np.asarray(clip_limits_6, dtype=np.float32).reshape(6)
        seq[:, :6] = np.clip(seq[:, :6], -clip_limits_6[None, :], clip_limits_6[None, :])

    if gripper_mode == "open":
        seq[:, 6] = -1.0
    elif gripper_mode == "close":
        seq[:, 6] = 1.0
    else:
        # Binary gripper command, held for several steps at a time.
        t = 0
        gripper_cmd = -1.0 if float(rng.random()) < 0.5 else 1.0
        while t < steps:
            hold = int(rng.integers(max(2, min_hold), max(max_hold, min_hold) + 1))
            next_t = min(steps, t + hold)
            seq[t:next_t, 6] = gripper_cmd
            t = next_t
            if float(rng.random()) < 0.7:
                gripper_cmd *= -1.0

    return seq


def summarize_action_sequence(actions: np.ndarray) -> dict:
    pos = actions[:, :3]
    rot = actions[:, 3:6]
    grip = actions[:, 6]
    return {
        "steps": int(actions.shape[0]),
        "pos_abs_max": np.abs(pos).max(axis=0).round(6).tolist(),
        "rot_abs_max": np.abs(rot).max(axis=0).round(6).tolist(),
        "gripper_unique": sorted(np.unique(grip).astype(np.float64).tolist()),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an empty LIBERO kitchen-table environment with only the robot gripper."
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Robosuite controller name, e.g. OSC_POSE or IK_POSE.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Render camera name when visualization is enabled.",
    )
    parser.add_argument(
        "--action-mode",
        type=str,
        default="mixed",
        choices=["mixed", "real", "random", "zero"],
        help="Action sequence type to execute.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display simulator frames in an OpenCV window, similar to WM/eval.py --show-sim-image.",
    )
    parser.add_argument("--sim-image-width", type=int, default=1280, help="Rendered simulator image width.")
    parser.add_argument("--sim-image-height", type=int, default=960, help="Rendered simulator image height.")
    parser.add_argument("--pos-limit", type=float, default=0.08, help="Max magnitude for each Cartesian action component.")
    parser.add_argument("--rot-limit", type=float, default=0.20, help="Max magnitude for each rotational action component.")
    parser.add_argument("--min-hold", type=int, default=8, help="Minimum number of steps to hold a target action segment.")
    parser.add_argument("--max-hold", type=int, default=20, help="Maximum number of steps to hold a target action segment.")
    parser.add_argument("--smoothing-alpha", type=float, default=0.72, help="EMA smoothing factor for the first 6 action dims.")
    parser.add_argument(
        "--gripper-mode",
        type=str,
        default="open",
        choices=["open", "close", "random"],
        help="How to set the binary gripper command during rollout.",
    )
    parser.add_argument(
        "--action-dataset-dir",
        action="append",
        default=None,
        help="Directory containing original LIBERO demo HDF5s for realistic action sampling. Can be passed multiple times.",
    )
    parser.add_argument(
        "--real-action-scale",
        type=float,
        default=0.70,
        help="Global multiplier for realistic action clips; lower values make the motion more conservative.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of steps to run after reset.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many episodes to run. Each episode resets the env and samples a new action sequence.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output HDF5 path. If provided, collected episodes are saved in _pcd.hdf5-compatible structure.",
    )
    parser.add_argument(
        "--latent-ckpt",
        type=str,
        default=None,
        help="Optional SDF latent checkpoint used to write meta/latent_mapping for vis_sdf_data.py compatibility.",
    )
    parser.add_argument(
        "--num-points-per-entity",
        type=int,
        default=256,
        help="Number of point samples for each gripper entity point cloud template.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Numpy / environment seed.",
    )
    parser.add_argument(
        "--print-obs-keys",
        action="store_true",
        help="Print all observation keys returned by the environment.",
    )
    parser.add_argument(
        "--save-bddl",
        type=str,
        default=None,
        help="Optional output path for the generated minimal BDDL file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    bddl_path = write_empty_kitchen_bddl(args.save_bddl)
    print(f"[INFO] BDDL: {bddl_path}")

    action_dataset_dirs = (
        [Path(p).expanduser() for p in args.action_dataset_dir]
        if args.action_dataset_dir
        else DEFAULT_ACTION_DATASET_DIRS
    )
    action_clips = None
    action_stats = None
    if args.action_mode in {"real", "mixed"}:
        action_clips, action_sources = load_action_clips(action_dataset_dirs, min_length=max(16, args.steps // 4))
        action_stats = compute_action_clip_stats(action_clips) if action_clips else None
        print(
            f"[INFO] Loaded {len(action_clips)} real action clips from "
            f"{len(list(iter_demo_hdf5_files(action_dataset_dirs)))} HDF5 files."
        )
        if not action_clips:
            raise RuntimeError(
                "No valid real action clips found. Pass --action-dataset-dir or switch to --action-mode random."
            )
        print(
            "[INFO] Conservative real-action envelope (p95 abs, first 6 dims): "
            f"{np.round(action_stats['p95_abs'], 4).tolist()}"
        )

    env = build_env(
        bddl_file_name=bddl_path,
        controller=args.controller,
        use_camera_obs=False,
        camera_name=args.camera,
    )

    try:
        env.seed(args.seed)
        gripper_spec, pose_extractor = build_gripper_spec_from_env(env, args.num_points_per_entity)
        latent_mapping = build_gripper_latent_mapping(gripper_spec["entities"], args.latent_ckpt)
        latent_pose_alignment = None
        if latent_mapping is not None:
            latent_mapping["latent_ckpt"] = str(Path(args.latent_ckpt).expanduser().resolve())
            latent_pose_alignment = {
                str(row["entity_name"]): {
                    "trans_local": np.asarray(row["pose_offset_local"], dtype=np.float32).reshape(3),
                    "rot_wxyz": np.asarray(row["pose_rot_offset_wxyz"], dtype=np.float32).reshape(4),
                }
                for row in latent_mapping["rows"]
            }
        window_name = "Gripper-Only Sim Image"
        if args.show:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, int(args.sim_image_width), int(args.sim_image_height))

        output_path = None
        if args.output:
            output_path = initialize_gripper_only_hdf5(
                output_path=args.output,
                bddl_path=bddl_path,
                env=env,
                latent_mapping=latent_mapping,
            )

        for episode_idx in range(max(1, int(args.episodes))):
            obs = safe_reset(env)
            summary = extract_env_summary(env, obs)
            if episode_idx == 0:
                print_env_summary(summary, obs, args.print_obs_keys)
            else:
                print(f"[INFO] Episode {episode_idx + 1}/{int(args.episodes)} reset.")
                print(
                    json.dumps(
                        {
                            "ee_pos": summary["ee_pos"],
                            "ee_quat": summary["ee_quat"],
                            "gripper_qpos": summary["gripper_qpos"],
                        },
                        indent=2,
                    )
                )

            if args.action_mode == "zero":
                actions = np.zeros((max(1, int(args.steps)), env.action_dim), dtype=np.float32)
                if args.gripper_mode == "open":
                    actions[:, 6] = -1.0
                elif args.gripper_mode == "close":
                    actions[:, 6] = 1.0
                sampled_mode = "zero"
            elif args.action_mode == "mixed":
                actions, sampled_mode = sample_mixed_action_sequence(
                    steps=args.steps,
                    action_dim=env.action_dim,
                    rng=rng,
                    action_clips=action_clips,
                    action_stats=action_stats,
                    gripper_mode=args.gripper_mode,
                    real_action_scale=args.real_action_scale,
                )
            elif args.action_mode == "real":
                actions = sample_realistic_action_sequence(
                    steps=args.steps,
                    action_dim=env.action_dim,
                    rng=rng,
                    action_clips=action_clips,
                    action_stats=action_stats,
                    gripper_mode=args.gripper_mode,
                    real_action_scale=args.real_action_scale,
                )
                sampled_mode = "real"
            else:
                actions = sample_smooth_action_sequence(
                    steps=args.steps,
                    action_dim=env.action_dim,
                    rng=rng,
                    pos_limit=args.pos_limit,
                    rot_limit=args.rot_limit,
                    min_hold=args.min_hold,
                    max_hold=args.max_hold,
                    smoothing_alpha=args.smoothing_alpha,
                    gripper_mode=args.gripper_mode,
                )
                sampled_mode = "random"

            summary_json = summarize_action_sequence(actions)
            summary_json["sampled_mode"] = sampled_mode
            print(f"[INFO] Episode {episode_idx + 1}/{int(args.episodes)} action sequence summary:")
            print(json.dumps(summary_json, indent=2))

            ep_data = collect_episode_data(
                env,
                actions=actions,
                gripper_spec=gripper_spec,
                pose_extractor=pose_extractor,
                latent_pose_alignment=latent_pose_alignment,
                show=args.show,
                camera_name=args.camera,
                sim_image_width=args.sim_image_width,
                sim_image_height=args.sim_image_height,
                window_name=window_name,
            )
            if output_path:
                append_gripper_only_episode(output_path, episode_idx, ep_data)
                if (episode_idx + 1) % 50 == 0:
                    print(f"[INFO] Saved {episode_idx + 1}/{int(args.episodes)} episodes to disk.")

        if output_path:
            print(f"[INFO] Saved gripper-only dataset to: {Path(output_path).expanduser().resolve()}")
        elif args.latent_ckpt:
            print("[WARN] --latent-ckpt was provided but no --output path was given, so no HDF5 file was written.")

        if args.show:
            cv2.destroyWindow(window_name)
    finally:
        env.close()


if __name__ == "__main__":
    main()
