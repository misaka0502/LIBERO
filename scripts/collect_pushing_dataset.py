from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import robosuite
import robosuite.macros as macros
import robosuite.utils.transform_utils as T
from robosuite import load_controller_config
from scipy.spatial.transform import Rotation as R

import init_path  # noqa: F401
import libero.libero.envs.bddl_utils as BDDLUtils
import libero.libero.utils.utils as libero_utils
from libero.libero.envs import TASK_MAPPING


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BDDL_BASE_PATH = REPO_ROOT / "LIBERO" / "libero" / "libero" / "bddl_files"
WM_ROOT = REPO_ROOT / "WM"
if str(WM_ROOT) not in sys.path:
    sys.path.append(str(WM_ROOT))

from omega.omega7_expert import CollectEnum, Omega7Expert  # noqa: E402


DEFAULT_SOURCE = (
    "/home/franka-client/datasets/libero/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)


@dataclass(frozen=True)
class EpisodeSeed:
    demo_key: str
    model_xml: Optional[str]
    init_state: Optional[np.ndarray]


@dataclass
class EpisodeBuffer:
    init_state: np.ndarray
    model_xml: str
    actions: List[np.ndarray]
    states: List[np.ndarray]
    robot_states: List[np.ndarray]
    gripper_states: List[np.ndarray]
    joint_states: List[np.ndarray]
    ee_states: List[np.ndarray]
    rewards: List[float]

    @classmethod
    def start(cls, env) -> "EpisodeBuffer":
        return cls(
            init_state=np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy(),
            model_xml=str(env.sim.model.get_xml()),
            actions=[],
            states=[],
            robot_states=[],
            gripper_states=[],
            joint_states=[],
            ee_states=[],
            rewards=[],
        )

    @property
    def num_steps(self) -> int:
        return len(self.actions)


class TerminalKeyReader:
    def __init__(self) -> None:
        self.enabled = False
        self.fd: Optional[int] = None
        self.old_attrs = None

    def __enter__(self) -> "TerminalKeyReader":
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old_attrs = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.enabled = True
        else:
            print("[WARN] stdin is not a TTY; keyboard controls will not be available.")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self.fd is not None and self.old_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)

    def poll(self) -> List[str]:
        if not self.enabled:
            return []
        keys: List[str] = []
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            ch = os.read(sys.stdin.fileno(), 1).decode("utf-8", errors="ignore")
            if ch:
                keys.append(ch)
        return keys


class ViewerKeyReader:
    def __init__(self, env) -> None:
        self.env = env
        self.enabled = False
        self._installed_viewer = None
        self._keys: List[str] = []

    def install(self) -> bool:
        viewer = getattr(self.env, "viewer", None)
        if self.enabled and viewer is self._installed_viewer:
            return False
        add_keypress_callback = getattr(viewer, "add_keypress_callback", None)
        if not callable(add_keypress_callback):
            return False

        try:
            import glfw

            key_map = {
                int(glfw.KEY_SPACE): " ",
                int(glfw.KEY_R): "r",
                int(glfw.KEY_Q): "q",
                int(glfw.KEY_ESCAPE): "\x1b",
            }
        except Exception:
            key_map = {
                32: " ",
                82: "r",
                81: "q",
                256: "\x1b",
            }
        cv2_key_map = {
            32: " ",
            ord("r"): "r",
            ord("R"): "r",
            ord("q"): "q",
            ord("Q"): "q",
            27: "\x1b",
        }

        def on_press(_window, key, _scancode, _action, _mods) -> None:
            mapped = key_map.get(int(key))
            if mapped is not None:
                self._keys.append(mapped)

        def on_cv2_key(key) -> None:
            key = int(key)
            if key < 0:
                return
            mapped = cv2_key_map.get(key & 0xFF, cv2_key_map.get(key))
            if mapped is not None:
                self._keys.append(mapped)

        try:
            for key_code in key_map:
                add_keypress_callback(key_code, on_press)
        except TypeError:
            add_keypress_callback(on_cv2_key)
        self.enabled = True
        self._installed_viewer = viewer
        return True

    def poll(self) -> List[str]:
        if not self._keys:
            return []
        keys = list(self._keys)
        self._keys.clear()
        return keys


class DualCameraOpenCVViewer:
    def __init__(
        self,
        base_viewer,
        env,
        camera_names: Sequence[str],
        width: int = 640,
        height: int = 480,
        window_name: str = "LIBERO teleop: agentview | wrist",
    ) -> None:
        self.base_viewer = base_viewer
        self.env = env
        self.width = int(width)
        self.height = int(height)
        self.window_name = str(window_name)
        self.keypress_callback = None
        self.camera_names = list(camera_names)

    def render(self) -> None:
        import cv2

        frames = []
        for camera_name in self.camera_names:
            frame = self.env.sim.render(
                camera_name=str(camera_name),
                height=self.height,
                width=self.width,
            )[..., ::-1]
            frame = np.flip(frame, axis=0)
            frames.append(frame)
        if not frames:
            return
        image = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
        cv2.imshow(self.window_name, image)
        key = cv2.waitKey(1)
        if self.keypress_callback is not None:
            self.keypress_callback(key)

    def add_keypress_callback(self, keypress_callback) -> None:
        self.keypress_callback = keypress_callback

    def reset(self) -> None:
        return None

    def close(self) -> None:
        try:
            import cv2

            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        close = getattr(self.base_viewer, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str):
        return getattr(self.base_viewer, name)


def _camera_exists(env, camera_name: str) -> bool:
    try:
        env.sim.model.camera_name2id(str(camera_name))
        return True
    except Exception:
        return False


def _available_camera_names(env) -> List[str]:
    names = getattr(env.sim.model, "camera_names", None)
    if names is None:
        return []
    return [str(name) for name in names]


def _resolve_render_cameras(env, requested: Sequence[str]) -> List[str]:
    available = _available_camera_names(env)
    resolved: List[str] = []
    for name in requested:
        if _camera_exists(env, name) and name not in resolved:
            resolved.append(str(name))

    if not resolved:
        for fallback in ("agentview", "frontview", "canonical_agentview"):
            if _camera_exists(env, fallback):
                resolved.append(fallback)
                break
    if len(resolved) < 2:
        for fallback in ("robot0_eye_in_hand", "eye_in_hand"):
            if _camera_exists(env, fallback) and fallback not in resolved:
                resolved.append(fallback)
                break
    if len(resolved) < 2:
        for name in available:
            if "eye_in_hand" in name and name not in resolved:
                resolved.append(name)
                break
    if len(resolved) < 2:
        for name in available:
            if name not in resolved:
                resolved.append(name)
                break
    return resolved[:2]


def _install_dual_camera_viewer(env, args: argparse.Namespace) -> bool:
    if bool(args.no_render) or bool(args.single_view):
        return False
    viewer = getattr(env, "viewer", None)
    if viewer is None or isinstance(viewer, DualCameraOpenCVViewer):
        return isinstance(viewer, DualCameraOpenCVViewer)
    cameras = _resolve_render_cameras(env, [args.main_camera, args.wrist_camera])
    if len(cameras) < 2:
        print(f"[WARN] dual-camera viewer unavailable; cameras found={_available_camera_names(env)}")
        return False
    env.viewer = DualCameraOpenCVViewer(
        base_viewer=viewer,
        env=env,
        camera_names=cameras,
        width=int(args.render_width),
        height=int(args.render_height),
    )
    print(f"[INFO] dual-camera viewer enabled: {' | '.join(cameras)}")
    return True


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _decode_attr_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="ignore")
    return str(value)


def _sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    keys = [str(k) for k in data_group.keys() if str(k).startswith("demo_")]
    return sorted(keys, key=lambda item: int(item.split("_", 1)[1]) if item.split("_", 1)[1].isdigit() else item)


def _safe_reset(env, max_tries: int = 10):
    last_error = None
    for _ in range(max_tries):
        try:
            return env.reset()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"Failed to reset environment after {max_tries} attempts") from last_error


def _rewrite_demo_model_xml_paths(xml_str: str) -> str:
    lib_assets_root = (REPO_ROOT / "LIBERO" / "libero" / "libero" / "assets").resolve()
    robosuite_root = Path(robosuite.__file__).resolve().parent
    tree = ET.fromstring(xml_str)
    asset = tree.find("asset")
    if asset is None:
        return xml_str

    for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
        old_path = elem.get("file")
        if not old_path:
            continue
        parts = old_path.split("/")
        new_path: Optional[Path] = None
        if "robosuite" in parts:
            ind = max(i for i, value in enumerate(parts) if value == "robosuite")
            new_path = robosuite_root.joinpath(*parts[ind + 1 :])
        elif "assets" in parts:
            ind = max(i for i, value in enumerate(parts) if value == "assets")
            new_path = lib_assets_root.joinpath(*parts[ind + 1 :])
        if new_path is not None:
            elem.set("file", str(new_path))

    return ET.tostring(tree, encoding="unicode")


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q = (q / n).astype(np.float32, copy=False)
    if q[0] < 0:
        q = (-q).astype(np.float32, copy=False)
    return q


def _quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(quat)
    return np.asarray([q[1], q[2], q[3], q[0]], dtype=np.float32)


def _snapshot_controller_ee_pose(env) -> np.ndarray:
    robots = getattr(env, "robots", None)
    if not robots:
        raise RuntimeError("Could not access env.robots for controller EE pose.")
    controller = getattr(robots[0], "controller", None)
    if controller is None:
        raise RuntimeError("Could not access robot controller for EE pose.")
    controller.update(force=True)
    ee_pos = np.asarray(controller.ee_pos, dtype=np.float32).reshape(3)
    ee_rot = np.asarray(controller.ee_ori_mat, dtype=np.float32).reshape(3, 3)
    ee_quat = _normalize_quat_wxyz(_rotmat_to_quat_wxyz(ee_rot))
    return np.concatenate([ee_pos, ee_quat], axis=0).astype(np.float32, copy=False)


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(np.asarray(rot, dtype=np.float64).reshape(3, 3)).as_quat().astype(np.float32)
    quat_wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    return _normalize_quat_wxyz(quat_wxyz)


def _quat_to_rotmat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(quat)
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().astype(np.float32)


def find_bddl_file(bddl_file_name: str) -> Optional[str]:
    if not bddl_file_name:
        return None

    parts = str(bddl_file_name).split("/")
    subdir = parts[-2] if len(parts) >= 2 else ""
    basename = parts[-1]

    exact = (BDDL_BASE_PATH / subdir / basename) if subdir else (BDDL_BASE_PATH / basename)
    if exact.exists():
        return str(exact)

    search_dir = (BDDL_BASE_PATH / subdir) if subdir else BDDL_BASE_PATH
    if not search_dir.exists():
        matches = list(BDDL_BASE_PATH.rglob(basename))
        if matches:
            return str(matches[0])
        search_dir = BDDL_BASE_PATH

    basename_no_ext = Path(basename).stem
    for path in search_dir.rglob("*.bddl"):
        if path.stem == basename_no_ext:
            return str(path)

    tokens = set(basename_no_ext.split("_"))
    best_path = None
    best_score = -1
    for path in search_dir.rglob("*.bddl"):
        score = len(tokens.intersection(set(path.stem.split("_"))))
        if score > best_score:
            best_score = score
            best_path = path
    return str(best_path) if best_path is not None else None


def _controller_output_scales(controller_cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    output_max = np.asarray(
        controller_cfg.get("output_max", (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)),
        dtype=np.float32,
    ).reshape(-1)
    output_min = np.asarray(
        controller_cfg.get("output_min", (-0.05, -0.05, -0.05, -0.5, -0.5, -0.5)),
        dtype=np.float32,
    ).reshape(-1)
    if output_max.shape[0] < 6 or output_min.shape[0] < 6:
        raise RuntimeError("controller_configs.output_{min,max} must contain at least 6 pose dimensions.")
    pos_scale = np.maximum(np.abs(output_max[:3]), np.abs(output_min[:3]))
    rot_scale = np.maximum(np.abs(output_max[3:6]), np.abs(output_min[3:6]))
    return np.maximum(pos_scale, 1e-6), np.maximum(rot_scale, 1e-6)


def _omega_absolute_target_to_env_delta_action(
    target_action: np.ndarray,
    current_hand_pose_wxyz: np.ndarray,
    controller_cfg: dict,
    env_action_dim: int,
) -> np.ndarray:
    target_action = np.asarray(target_action, dtype=np.float32).reshape(-1)
    if target_action.shape[0] < 8:
        raise RuntimeError(f"Omega target action must be [x,y,z,qx,qy,qz,qw,grasp], got {target_action.shape}.")
    current_hand_pose_wxyz = np.asarray(current_hand_pose_wxyz, dtype=np.float32).reshape(7)
    pos_scale, rot_scale = _controller_output_scales(controller_cfg)

    cur_pos = current_hand_pose_wxyz[:3]
    cur_rot = _quat_to_rotmat_wxyz(current_hand_pose_wxyz[3:7]).astype(np.float32, copy=False)
    target_pos = target_action[:3].astype(np.float32, copy=False)
    target_rot = R.from_quat(target_action[3:7].astype(np.float64)).as_matrix().astype(np.float32, copy=False)

    delta_pos = target_pos - cur_pos
    delta_rot = target_rot @ cur_rot.T
    delta_rotvec = R.from_matrix(delta_rot.astype(np.float64)).as_rotvec().astype(np.float32, copy=False)

    delta_pos_cmd = np.clip(delta_pos / pos_scale, -1.0, 1.0).astype(np.float32, copy=False)
    delta_rot_cmd = np.clip(delta_rotvec / rot_scale, -1.0, 1.0).astype(np.float32, copy=False)
    grasp_cmd = np.clip(target_action[7:8].astype(np.float32, copy=False), -1.0, 1.0)

    if int(env_action_dim) == 7:
        return np.concatenate([delta_pos_cmd, delta_rot_cmd, grasp_cmd], axis=0).astype(np.float32, copy=False)
    if int(env_action_dim) == 6:
        return np.concatenate([delta_pos_cmd, delta_rot_cmd], axis=0).astype(np.float32, copy=False)
    raise RuntimeError(f"Unsupported env.action_dim={env_action_dim}; expected 6 or 7.")


def _create_omega_for_current_ee(env, args: argparse.Namespace) -> Omega7Expert:
    ee_pose_wxyz = _snapshot_controller_ee_pose(env)
    return Omega7Expert(
        control_mode="pos",
        robot_workspace_center=ee_pose_wxyz[:3].astype(np.float32, copy=False),
        robot_init_quat=_quat_wxyz_to_xyzw(ee_pose_wxyz[3:7]),
        pos_sensitivity=float(args.omega_pos_sensitivity),
        rot_sensitivity=float(args.omega_rot_sensitivity),
        workspace_radius=float(args.omega_workspace_radius),
        dead_zone_ratio=float(args.omega_dead_zone_ratio),
        use_quat=True,
        free_start=True,
    )


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def _read_data_attrs(path: Path) -> Dict[str, object]:
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"Missing data group in {path}")
        return {str(k): v for k, v in f["data"].attrs.items()}


def _initialize_output_hdf5(
    output_path: Path,
    template_hdf5: Optional[Path],
    env_name: str,
    problem_info: dict,
    env_args: dict,
    bddl_path: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        with h5py.File(output_path, "a") as f:
            if "data" not in f:
                raise RuntimeError(f"Existing output has no data group: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as out_f:
        data_group = out_f.create_group("data")
        if template_hdf5 is not None and template_hdf5.exists():
            with h5py.File(template_hdf5, "r") as template_f:
                if "data" in template_f:
                    _copy_attrs(template_f["data"].attrs, data_group.attrs)
        data_group.attrs["env_name"] = str(env_name)
        data_group.attrs["problem_info"] = json.dumps(_jsonable(problem_info))
        data_group.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION
        data_group.attrs["env_args"] = json.dumps(_jsonable(env_args))
        data_group.attrs["bddl_file_name"] = str(bddl_path)
        try:
            data_group.attrs["bddl_file_content"] = Path(bddl_path).read_text(encoding="utf-8")
        except Exception:
            pass
        data_group.attrs["tag"] = str(data_group.attrs.get("tag", "libero-v1"))
        data_group.attrs["num_demos"] = 0
        data_group.attrs["total"] = 0


def _make_hdf5_env_config(source_path: Path, args: argparse.Namespace) -> Tuple[dict, dict, str]:
    attrs = _read_data_attrs(source_path)
    env_args_raw = attrs.get("env_args")
    if env_args_raw is None:
        raise RuntimeError(f"{source_path} is missing data.attrs['env_args']")
    env_args = json.loads(_decode_attr_text(env_args_raw))
    env_kwargs = dict(env_args.get("env_kwargs", {}) or {})
    problem_name = str(env_args.get("problem_name", "") or "")
    if not problem_name:
        problem_info_raw = attrs.get("problem_info", "{}")
        problem_info = json.loads(_decode_attr_text(problem_info_raw))
        problem_name = str(problem_info.get("problem_name", ""))
    if problem_name not in TASK_MAPPING:
        raise RuntimeError(f"Unsupported or missing LIBERO problem_name={problem_name!r}")

    bddl_file_name = str(attrs.get("bddl_file_name", "") or env_kwargs.get("bddl_file_name", ""))
    resolved_bddl = find_bddl_file(bddl_file_name)
    if not resolved_bddl:
        raise RuntimeError(f"Could not resolve BDDL path from {bddl_file_name!r}")

    env_kwargs["bddl_file_name"] = str(resolved_bddl)
    env_kwargs["has_renderer"] = not bool(args.no_render)
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs["camera_depths"] = False
    env_kwargs["ignore_done"] = True
    env_kwargs["reward_shaping"] = True
    env_kwargs.setdefault("control_freq", 20)
    env_kwargs.setdefault("robots", ["Panda"])
    if "controller_configs" not in env_kwargs:
        env_kwargs["controller_configs"] = load_controller_config(default_controller="OSC_POSE")

    env_args = dict(env_args)
    env_args["problem_name"] = problem_name
    env_args["env_name"] = str(attrs.get("env_name", env_args.get("env_name", "")))
    env_args["bddl_file"] = str(resolved_bddl)
    env_args["env_kwargs"] = _jsonable(env_kwargs)
    return env_args, env_kwargs, str(resolved_bddl)


def _make_bddl_env_config(source_path: Path, args: argparse.Namespace) -> Tuple[dict, dict, str, dict]:
    problem_info = BDDLUtils.get_problem_info(str(source_path))
    problem_name = str(problem_info["problem_name"])
    if problem_name not in TASK_MAPPING:
        raise RuntimeError(f"Unsupported LIBERO problem_name={problem_name!r}")
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env_kwargs = {
        "bddl_file_name": str(source_path),
        "robots": ["Panda"],
        "controller_configs": controller_config,
        "has_renderer": not bool(args.no_render),
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "ignore_done": True,
        "reward_shaping": True,
        "control_freq": 20,
    }
    env_args = {
        "type": 1,
        "env_name": "",
        "problem_name": problem_name,
        "bddl_file": str(source_path),
        "env_kwargs": _jsonable(env_kwargs),
    }
    return env_args, env_kwargs, str(source_path), problem_info


def _load_problem_info_from_template(template_hdf5: Path, fallback_bddl: str) -> dict:
    with h5py.File(template_hdf5, "r") as f:
        raw = f.get("data").attrs.get("problem_info", None) if "data" in f else None
    if raw is not None:
        try:
            return json.loads(_decode_attr_text(raw))
        except Exception:
            pass
    return BDDLUtils.get_problem_info(fallback_bddl)


def _load_episode_seed(source_hdf5: Path, demo_key: str) -> EpisodeSeed:
    with h5py.File(source_hdf5, "r") as f:
        demo_group = f[f"data/{demo_key}"]
        model_xml = demo_group.attrs.get("model_file", None)
        model_xml = _decode_attr_text(model_xml) if model_xml is not None else None
        init_state = demo_group.attrs.get("init_state", None)
        if init_state is None and "states" in demo_group and demo_group["states"].shape[0] > 0:
            init_state = np.asarray(demo_group["states"][0], dtype=np.float64)
        init_state_arr = None if init_state is None else np.asarray(init_state, dtype=np.float64).reshape(-1)
        return EpisodeSeed(demo_key=demo_key, model_xml=model_xml, init_state=init_state_arr)


def _reset_to_episode_seed(env, seed: EpisodeSeed) -> None:
    _safe_reset(env)
    if seed.model_xml:
        model_xml = libero_utils.postprocess_model_xml(seed.model_xml, {})
        model_xml = _rewrite_demo_model_xml_paths(model_xml)
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
    if seed.init_state is not None:
        env.sim.set_state_from_flattened(seed.init_state)
    env.sim.forward()


def _reset_from_bddl(env) -> EpisodeSeed:
    _safe_reset(env)
    env.sim.forward()
    return EpisodeSeed(
        demo_key="bddl_reset",
        model_xml=str(env.sim.model.get_xml()),
        init_state=np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy(),
    )


def _append_episode(
    output_path: Path,
    buffer: EpisodeBuffer,
) -> Tuple[str, int]:
    if buffer.num_steps <= 0:
        raise RuntimeError("Cannot append an empty episode.")

    actions = np.asarray(buffer.actions, dtype=np.float64)
    states = np.asarray(buffer.states, dtype=np.float64)
    robot_states = np.asarray(buffer.robot_states, dtype=np.float64) if buffer.robot_states else None
    gripper_states = np.asarray(buffer.gripper_states, dtype=np.float64) if buffer.gripper_states else None
    joint_states = np.asarray(buffer.joint_states, dtype=np.float64) if buffer.joint_states else None
    ee_states = np.asarray(buffer.ee_states, dtype=np.float64) if buffer.ee_states else None
    dones = np.zeros((buffer.num_steps,), dtype=np.uint8)
    dones[-1] = 1
    rewards = np.zeros((buffer.num_steps,), dtype=np.uint8)
    rewards[-1] = 1

    with h5py.File(output_path, "a") as f:
        data_group = f["data"]
        ep_idx = int(data_group.attrs.get("num_demos", 0))
        demo_key = f"demo_{ep_idx}"
        while demo_key in data_group:
            ep_idx += 1
            demo_key = f"demo_{ep_idx}"

        demo_group = data_group.create_group(demo_key)
        demo_group.attrs["init_state"] = np.asarray(buffer.init_state, dtype=np.float64)
        demo_group.attrs["model_file"] = str(buffer.model_xml)
        demo_group.attrs["num_samples"] = int(buffer.num_steps)

        demo_group.create_dataset("actions", data=actions)
        demo_group.create_dataset("states", data=states)
        demo_group.create_dataset("rewards", data=rewards)
        demo_group.create_dataset("dones", data=dones)
        if robot_states is not None:
            demo_group.create_dataset("robot_states", data=robot_states)

        obs_group = demo_group.create_group("obs")
        if gripper_states is not None:
            obs_group.create_dataset("gripper_states", data=gripper_states)
        if joint_states is not None:
            obs_group.create_dataset("joint_states", data=joint_states)
        if ee_states is not None:
            obs_group.create_dataset("ee_states", data=ee_states)
            obs_group.create_dataset("ee_pos", data=ee_states[:, :3])
            obs_group.create_dataset("ee_ori", data=ee_states[:, 3:])

        data_group.attrs["num_demos"] = int(data_group.attrs.get("num_demos", 0)) + 1
        data_group.attrs["total"] = int(data_group.attrs.get("total", 0)) + int(buffer.num_steps)
        return demo_key, int(buffer.num_steps)


def _record_step(env, action: np.ndarray, buffer: EpisodeBuffer) -> None:
    state_flat = np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
    obs, reward, _done, _info = env.step(action)

    buffer.states.append(state_flat)
    buffer.actions.append(np.asarray(action, dtype=np.float64).copy())
    buffer.rewards.append(float(reward))

    try:
        buffer.robot_states.append(np.asarray(env.get_robot_state_vector(obs), dtype=np.float64).copy())
    except Exception:
        pass
    if "robot0_gripper_qpos" in obs:
        buffer.gripper_states.append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).copy())
    if "robot0_joint_pos" in obs:
        buffer.joint_states.append(np.asarray(obs["robot0_joint_pos"], dtype=np.float64).copy())
    if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
        ee_state = np.hstack(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                T.quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
            )
        )
        buffer.ee_states.append(np.asarray(ee_state, dtype=np.float64).copy())


def _print_status(
    phase: str,
    saved: int,
    discarded: int,
    current_steps: int,
    last_len: int,
    output_path: Path,
) -> None:
    print(
        f"[{phase}] saved={saved} discarded={discarded} current_steps={current_steps} "
        f"last_len={last_len} output={output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect LIBERO pushing trajectories with Omega.7 teleoperation.")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Input .hdf5 or .bddl scene source.")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 path.")
    parser.add_argument(
        "--template-hdf5",
        type=str,
        default=None,
        help="Optional HDF5 used only for copying data attrs when --source is a .bddl file.",
    )
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--omega-pos-sensitivity", type=float, default=5.0)
    parser.add_argument("--omega-rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--omega-workspace-radius", type=float, default=0.5)
    parser.add_argument("--omega-dead-zone-ratio", type=float, default=0.9)
    parser.add_argument("--no-render", action="store_true", help="Disable robosuite on-screen rendering.")
    parser.add_argument("--single-view", action="store_true", help="Use robosuite's default single-camera viewer.")
    parser.add_argument("--main-camera", type=str, default="agentview", help="Main camera for the stitched viewer.")
    parser.add_argument(
        "--wrist-camera",
        type=str,
        default="robot0_eye_in_hand",
        help="Wrist camera for the stitched viewer.",
    )
    parser.add_argument("--render-width", type=int, default=640, help="Per-camera viewer width.")
    parser.add_argument("--render-height", type=int, default=480, help="Per-camera viewer height.")
    parser.add_argument("--overwrite", action="store_true", help="Reinitialize output HDF5 before collecting.")
    parser.add_argument(
        "--dry-run-init",
        action="store_true",
        help="Initialize one scene, then exit without creating output or touching Omega.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source does not exist: {source_path}")

    is_hdf5 = source_path.suffix.lower() in {".hdf5", ".h5"}
    is_bddl = source_path.suffix.lower() == ".bddl"
    if not (is_hdf5 or is_bddl):
        raise ValueError(f"--source must be .hdf5 or .bddl, got: {source_path}")

    rng = np.random.default_rng(int(args.seed))
    template_hdf5: Optional[Path]
    if is_hdf5:
        template_hdf5 = source_path
    elif args.template_hdf5 is not None:
        template_hdf5 = Path(args.template_hdf5).expanduser().resolve()
        if not template_hdf5.exists():
            raise FileNotFoundError(f"Template HDF5 does not exist: {template_hdf5}")
    else:
        template_hdf5 = None

    if is_hdf5:
        env_args, env_kwargs, bddl_path = _make_hdf5_env_config(source_path, args)
        problem_info = _load_problem_info_from_template(source_path, bddl_path)
        demo_keys: List[str]
        with h5py.File(source_path, "r") as f:
            demo_keys = _sorted_demo_keys(f["data"])
        if not demo_keys:
            raise RuntimeError(f"No demo_* groups found in source HDF5: {source_path}")
    else:
        env_args, env_kwargs, bddl_path, problem_info = _make_bddl_env_config(source_path, args)
        demo_keys = []

    env = TASK_MAPPING[str(env_args["problem_name"])](**env_kwargs)
    env_args["env_name"] = env.__class__.__name__
    if bool(args.dry_run_init):
        try:
            if is_hdf5:
                demo_key = str(rng.choice(demo_keys))
                seed = _load_episode_seed(source_path, demo_key)
                _reset_to_episode_seed(env, seed)
                print(f"[DRY-RUN] restored {demo_key}")
            else:
                _reset_from_bddl(env)
                print("[DRY-RUN] restored BDDL scene")
            print(
                f"[DRY-RUN] env={env.__class__.__name__} action_dim={getattr(env, 'action_dim', None)} "
                "raw_demo_schema=actions/states/rewards/dones/robot_states/obs_without_images"
            )
            print("[DRY-RUN] no output written; Omega was not initialized")
            return
        finally:
            env.close()

    _initialize_output_hdf5(
        output_path=output_path,
        template_hdf5=template_hdf5,
        env_name=env.__class__.__name__,
        problem_info=problem_info,
        env_args=env_args,
        bddl_path=bddl_path,
        overwrite=bool(args.overwrite),
    )
    if int(args.max_episodes) <= 0:
        env.close()
        print(f"[INFO] initialized output only: {output_path}")
        return
    omega: Optional[Omega7Expert] = None
    buffer: Optional[EpisodeBuffer] = None
    saved = 0
    discarded = 0
    last_len = 0
    last_status_time = 0.0
    phase = "READY"
    update_period = 1.0 / max(float(args.control_hz), 1e-3)
    last_step_time = time.monotonic()

    def reset_round() -> None:
        nonlocal omega, buffer, phase, last_step_time
        if omega is not None:
            omega.close()
            omega = None
        if is_hdf5:
            demo_key = str(rng.choice(demo_keys))
            seed = _load_episode_seed(source_path, demo_key)
            _reset_to_episode_seed(env, seed)
            print(f"[RESET] restored {demo_key}")
        else:
            seed = _reset_from_bddl(env)
            print("[RESET] restored BDDL scene")
        omega = _create_omega_for_current_ee(env, args)
        buffer = None
        phase = "READY"
        last_step_time = time.monotonic()

    try:
        reset_round()
        _install_dual_camera_viewer(env, args)
        viewer_keys = ViewerKeyReader(env)
        viewer_key_notice_printed = False

        def ensure_viewer_keys() -> None:
            nonlocal viewer_key_notice_printed
            if args.no_render:
                return
            installed = viewer_keys.install()
            if installed and not viewer_key_notice_printed:
                print("[INFO] viewer keyboard controls enabled")
                viewer_key_notice_printed = True

        def prepare_viewer() -> None:
            if args.no_render:
                return
            _install_dual_camera_viewer(env, args)
            ensure_viewer_keys()

        prepare_viewer()
        print("[INFO] controls: SPACE=start/stop recording | r=discard/reset | q or ESC=quit")
        with TerminalKeyReader() as keys:
            while saved < int(args.max_episodes):
                now = time.monotonic()
                for key in keys.poll() + viewer_keys.poll():
                    if key in {"q", "\x1b"}:
                        print("[INFO] quit requested")
                        return
                    if key == " ":
                        if phase == "READY":
                            if omega is not None:
                                omega.set_current_pose_as_origin()
                            buffer = EpisodeBuffer.start(env)
                            phase = "RECORDING"
                            last_step_time = time.monotonic()
                            print("[INFO] recording started; Omega origin set to current pose")
                        elif phase == "RECORDING" and buffer is not None:
                            if buffer.num_steps <= 0:
                                print("[WARN] empty trajectory ignored")
                                reset_round()
                                prepare_viewer()
                                continue
                            demo_key, last_len = _append_episode(output_path, buffer)
                            saved += 1
                            print(f"[SAVED] {demo_key} len={last_len}")
                            reset_round()
                            prepare_viewer()
                    elif key.lower() == "r":
                        if phase == "RECORDING" and buffer is not None and buffer.num_steps > 0:
                            discarded += 1
                            print(f"[DISCARD] len={buffer.num_steps}")
                        else:
                            print("[RESET] no active trajectory")
                        reset_round()
                        prepare_viewer()

                if not args.no_render:
                    prepare_viewer()
                    env.render()

                if phase == "RECORDING" and buffer is not None and omega is not None:
                    if buffer.num_steps >= int(args.max_steps):
                        demo_key, last_len = _append_episode(output_path, buffer)
                        saved += 1
                        print(f"[SAVED] {demo_key} len={last_len} reason=max_steps")
                        reset_round()
                        prepare_viewer()
                        continue
                    if (now - last_step_time) >= update_period:
                        target_action, enum = omega.get_action(transform=False)
                        if enum == CollectEnum.DONE_TRUE:
                            last_step_time = now
                            current_pose = _snapshot_controller_ee_pose(env)
                            action = _omega_absolute_target_to_env_delta_action(
                                target_action=np.asarray(target_action, dtype=np.float32),
                                current_hand_pose_wxyz=current_pose,
                                controller_cfg=dict(env_kwargs.get("controller_configs", {}) or {}),
                                env_action_dim=int(getattr(env, "action_dim", 0)),
                            )
                            _record_step(env, action, buffer)

                if (now - last_status_time) >= 1.0:
                    _print_status(
                        phase=phase,
                        saved=saved,
                        discarded=discarded,
                        current_steps=0 if buffer is None else buffer.num_steps,
                        last_len=last_len,
                        output_path=output_path,
                    )
                    last_status_time = now
                time.sleep(0.002)
    finally:
        if omega is not None:
            omega.close()
        env.close()


if __name__ == "__main__":
    main()
