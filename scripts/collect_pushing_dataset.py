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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


DEFAULT_AUTO_PUSH_CONFIG: Dict[str, Any] = {
    "pushes_per_episode": 4,
    "max_sampling_attempts": 80,
    "workspace_margin": 0.08,
    "object_radius_fallback": 0.05,
    "planning_radius_padding": 0.02,
    "ee_radius": 0.02,
    "contact_margin": 0.01,
    "descent_extra_clearance": 0.08,
    "edge_direction_candidates": 12,
    "edge_threshold_ratio": 0.72,
    "edge_direction_noise_degrees": 20.0,
    "clearance_height": 0.15,
    "z_push_range": [0.015, 0.04],
    "z_push_limits": [0.01, 0.06],
    "adaptive_z_push": {
        "enabled": True,
        "height_fraction_range": [0.20, 0.45],
        "radius_fraction_range": [0.18, 0.45],
        "min_offset": 0.008,
        "max_offset": 0.045,
    },
    "osc_kp": 5.0,
    "osc_max_delta": 0.03,
    "episode_speed_scale_range": [0.75, 1.35],
    "pos_tolerance": 0.005,
    "max_steps_per_waypoint": 80,
    "push_waypoints": 8,
    "push_steps_per_waypoint": 20,
    "settle_steps": 5,
    "boundary_threshold_ratio": 0.72,
    "target_sampling_weights": {
        "nearest_neighbor": 0.50,
        "least_pushed": 0.25,
        "uniform": 0.25,
    },
    "push_type_weights": {
        "object_to_object": 0.30,
        "random_free": 0.30,
        "grazing": 0.15,
        "cluster": 0.15,
        "boundary_recovery": 0.05,
        "near_miss_or_weak": 0.05,
    },
    "gripper_weights": {
        "closed": 0.50,
        "half_open": 0.30,
        "open": 0.20,
    },
    "gripper_commands": {
        "closed": -1.0,
        "half_open": 0.0,
        "open": 1.0,
    },
    "push_length_ranges": {
        "default": [0.08, 0.20],
        "weak": [0.03, 0.07],
    },
    "lateral_offset_ranges": {
        "default": [-0.5, 0.5],
        "grazing_abs": [0.5, 0.9],
        "near_miss_abs": [1.1, 1.5],
    },
    "angle_noise_degrees": {
        "object_to_object": 30.0,
        "cluster": 30.0,
    },
    "quality_filter": {
        "enabled": True,
        "no_contact_keep_probability": 0.03,
        "weak_contact_keep_probability": 0.15,
        "object_object_small_motion_keep_probability": 0.25,
        "min_keep_displacement": 0.02,
        "min_keep_rotation_deg": 10.0,
        "moved_object_displacement": 0.012,
        "moved_object_rotation_deg": 8.0,
        "significant_displacement": 0.04,
        "significant_rotation_deg": 15.0,
        "min_eef_object_contacts": 2,
        "max_object_z_delta": 0.05,
        "max_object_z_lift_during_episode": 0.035,
        "leave_table_margin": 0.02,
        "max_robot_table_contacts": 15,
    },
}


@dataclass(frozen=True)
class EpisodeSeed:
    demo_key: str
    model_xml: Optional[str]
    init_state: Optional[np.ndarray]


@dataclass(frozen=True)
class AutoObjectState:
    name: str
    pos: np.ndarray
    quat_wxyz: np.ndarray
    radius: float
    height: float

    @property
    def xy(self) -> np.ndarray:
        return self.pos[:2]


@dataclass(frozen=True)
class AutoPushParams:
    target_name: str
    push_type: str
    approach_xy: np.ndarray
    start_xy: np.ndarray
    end_xy: np.ndarray
    z_push: float
    gripper_cmd: float


@dataclass
class AutoEpisodeMetrics:
    initial_pos: Dict[str, np.ndarray]
    initial_quat: Dict[str, np.ndarray]
    max_z_lift: Dict[str, float]
    eef_object_contacts: int = 0
    object_object_contacts: int = 0
    robot_table_contacts: int = 0
    leaves_table: bool = False


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


def _deep_copy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(config))


def _deep_update_config(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update_config(base[key], value)
        else:
            base[key] = value
    return base


def _normalize_weight_dict(config: Dict[str, Any], key: str) -> Dict[str, float]:
    raw = config.get(key, {})
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError(f"Auto config field {key!r} must be a non-empty object.")
    weights = {str(k): float(v) for k, v in raw.items()}
    if any(v < 0.0 for v in weights.values()):
        raise RuntimeError(f"Auto config field {key!r} contains negative weights: {weights}")
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise RuntimeError(f"Auto config field {key!r} must have positive total weight.")
    return {k: v / total for k, v in weights.items()}


def _validate_auto_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config = _deep_copy_config(config)
    config["target_sampling_weights"] = _normalize_weight_dict(config, "target_sampling_weights")
    config["push_type_weights"] = _normalize_weight_dict(config, "push_type_weights")
    config["gripper_weights"] = _normalize_weight_dict(config, "gripper_weights")
    for key in (
        "pushes_per_episode",
        "max_sampling_attempts",
        "max_steps_per_waypoint",
        "push_waypoints",
        "push_steps_per_waypoint",
        "settle_steps",
        "edge_direction_candidates",
    ):
        config[key] = int(config[key])
        if config[key] < 0:
            raise RuntimeError(f"Auto config field {key!r} must be non-negative.")
    for key in (
        "workspace_margin",
        "object_radius_fallback",
        "planning_radius_padding",
        "ee_radius",
        "contact_margin",
        "descent_extra_clearance",
        "edge_threshold_ratio",
        "edge_direction_noise_degrees",
        "clearance_height",
        "osc_kp",
        "osc_max_delta",
        "pos_tolerance",
        "boundary_threshold_ratio",
    ):
        config[key] = float(config[key])
        if config[key] < 0.0:
            raise RuntimeError(f"Auto config field {key!r} must be non-negative.")
    if config["pushes_per_episode"] <= 0:
        raise RuntimeError("Auto config field 'pushes_per_episode' must be positive.")
    if config["max_sampling_attempts"] <= 0:
        raise RuntimeError("Auto config field 'max_sampling_attempts' must be positive.")
    if config["max_steps_per_waypoint"] <= 0:
        raise RuntimeError("Auto config field 'max_steps_per_waypoint' must be positive.")
    if config["push_waypoints"] <= 0:
        raise RuntimeError("Auto config field 'push_waypoints' must be positive.")
    if len(config.get("episode_speed_scale_range", [])) != 2:
        raise RuntimeError("Auto config field 'episode_speed_scale_range' must be a [min, max] range.")
    lo, hi = float(config["episode_speed_scale_range"][0]), float(config["episode_speed_scale_range"][1])
    if lo <= 0.0 or hi <= 0.0:
        raise RuntimeError("Auto config field 'episode_speed_scale_range' must contain positive values.")
    config["episode_speed_scale_range"] = [min(lo, hi), max(lo, hi)]
    qcfg = config.get("quality_filter", {})
    if not isinstance(qcfg, dict):
        raise RuntimeError("Auto config field 'quality_filter' must be an object.")
    qcfg["enabled"] = bool(qcfg.get("enabled", True))
    for key in (
        "no_contact_keep_probability",
        "weak_contact_keep_probability",
        "object_object_small_motion_keep_probability",
        "min_keep_displacement",
        "min_keep_rotation_deg",
        "moved_object_displacement",
        "moved_object_rotation_deg",
        "significant_displacement",
        "significant_rotation_deg",
        "max_object_z_delta",
        "max_object_z_lift_during_episode",
        "leave_table_margin",
    ):
        qcfg[key] = float(qcfg[key])
        if qcfg[key] < 0.0:
            raise RuntimeError(f"Auto config quality_filter field {key!r} must be non-negative.")
    for key in (
        "no_contact_keep_probability",
        "weak_contact_keep_probability",
        "object_object_small_motion_keep_probability",
    ):
        qcfg[key] = float(np.clip(qcfg[key], 0.0, 1.0))
    qcfg["min_eef_object_contacts"] = int(qcfg["min_eef_object_contacts"])
    qcfg["max_robot_table_contacts"] = int(qcfg["max_robot_table_contacts"])
    if qcfg["min_eef_object_contacts"] < 0 or qcfg["max_robot_table_contacts"] < 0:
        raise RuntimeError("Auto config quality_filter contact count thresholds must be non-negative.")
    config["quality_filter"] = qcfg
    zcfg = config.get("adaptive_z_push", {})
    if not isinstance(zcfg, dict):
        raise RuntimeError("Auto config field 'adaptive_z_push' must be an object.")
    zcfg["enabled"] = bool(zcfg.get("enabled", True))
    for key in ("height_fraction_range", "radius_fraction_range"):
        if len(zcfg.get(key, [])) != 2:
            raise RuntimeError(f"Auto config adaptive_z_push field {key!r} must be a [min, max] range.")
        lo, hi = float(zcfg[key][0]), float(zcfg[key][1])
        if lo < 0.0 or hi < 0.0:
            raise RuntimeError(f"Auto config adaptive_z_push field {key!r} must be non-negative.")
        zcfg[key] = [min(lo, hi), max(lo, hi)]
    for key in ("min_offset", "max_offset"):
        zcfg[key] = float(zcfg[key])
        if zcfg[key] < 0.0:
            raise RuntimeError(f"Auto config adaptive_z_push field {key!r} must be non-negative.")
    if zcfg["max_offset"] < zcfg["min_offset"]:
        zcfg["min_offset"], zcfg["max_offset"] = zcfg["max_offset"], zcfg["min_offset"]
    config["adaptive_z_push"] = zcfg
    return config


def _load_auto_config(path: Optional[str]) -> Dict[str, Any]:
    config = _deep_copy_config(DEFAULT_AUTO_PUSH_CONFIG)
    if path:
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as f:
            user_config = json.load(f)
        if not isinstance(user_config, dict):
            raise RuntimeError(f"Auto config must be a JSON object: {config_path}")
        _deep_update_config(config, user_config)
    return _validate_auto_config(config)


def _dump_auto_config(path: str) -> None:
    config_path = Path(path).expanduser().resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_AUTO_PUSH_CONFIG, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[INFO] wrote default auto config: {config_path}")


def _weighted_choice(weights: Dict[str, float], rng: np.random.Generator) -> str:
    keys = list(weights.keys())
    probs = np.asarray([float(weights[k]) for k in keys], dtype=np.float64)
    probs = probs / max(float(probs.sum()), 1e-12)
    return str(keys[int(rng.choice(len(keys), p=probs))])


def _sample_uniform_range(values: Sequence[float], rng: np.random.Generator) -> float:
    if len(values) != 2:
        raise RuntimeError(f"Expected a [min, max] range, got {values!r}")
    lo, hi = float(values[0]), float(values[1])
    if hi < lo:
        lo, hi = hi, lo
    return float(rng.uniform(lo, hi))


def _unit_vector(vec: np.ndarray) -> Optional[np.ndarray]:
    vec = np.asarray(vec, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return None
    return (vec / norm).astype(np.float64, copy=False)


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    return np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)


def _rotate_xy(vec: np.ndarray, degrees: float) -> np.ndarray:
    rad = np.deg2rad(float(degrees))
    c, s = float(np.cos(rad)), float(np.sin(rad))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return rot @ np.asarray(vec, dtype=np.float64).reshape(2)


def _table_full_size(env) -> np.ndarray:
    for attr in (
        "table_full_size",
        "kitchen_table_full_size",
        "study_table_full_size",
        "coffee_table_full_size",
        "living_room_table_full_size",
    ):
        if hasattr(env, attr):
            value = np.asarray(getattr(env, attr), dtype=np.float64).reshape(-1)
            if value.shape[0] >= 2:
                return value[:2]
    return np.asarray([1.0, 1.2], dtype=np.float64)


def _workspace_center(env) -> np.ndarray:
    offset = np.asarray(getattr(env, "workspace_offset", (0.0, 0.0, 0.9)), dtype=np.float64).reshape(-1)
    if offset.shape[0] < 2:
        return np.zeros(2, dtype=np.float64)
    return offset[:2]


def _table_height(env) -> float:
    offset = np.asarray(getattr(env, "workspace_offset", (0.0, 0.0, 0.9)), dtype=np.float64).reshape(-1)
    if offset.shape[0] >= 3:
        return float(offset[2])
    for attr in ("table_offset", "kitchen_table_offset", "study_table_offset", "coffee_table_offset", "living_room_table_offset"):
        if hasattr(env, attr):
            value = np.asarray(getattr(env, attr), dtype=np.float64).reshape(-1)
            if value.shape[0] >= 3:
                return float(value[2])
    return 0.9


def _workspace_bounds(env, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    center = _workspace_center(env)
    half_size = 0.5 * _table_full_size(env)
    margin = float(config["workspace_margin"])
    lower = center - np.maximum(half_size - margin, 0.02)
    upper = center + np.maximum(half_size - margin, 0.02)
    return lower.astype(np.float64), upper.astype(np.float64)


def _table_xy_bounds(env, extra_margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    center = _workspace_center(env)
    half_size = 0.5 * _table_full_size(env) + float(extra_margin)
    return (center - half_size).astype(np.float64), (center + half_size).astype(np.float64)


def _in_workspace(xy: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    xy = np.asarray(xy, dtype=np.float64).reshape(2)
    return bool(np.all(xy >= lower) and np.all(xy <= upper))


def _collect_body_subtree_ids(model, root_body_id: int) -> List[int]:
    body_parentid = np.asarray(getattr(model, "body_parentid", []), dtype=np.int64).reshape(-1)
    if body_parentid.size == 0:
        return [int(root_body_id)]
    result = [int(root_body_id)]
    queue = [int(root_body_id)]
    while queue:
        parent = queue.pop(0)
        children = np.where(body_parentid == parent)[0].astype(int).tolist()
        for child in children:
            if child not in result:
                result.append(child)
                queue.append(child)
    return result


def _estimate_object_extent(env, body_id: int, fallback_radius: float) -> Tuple[float, float]:
    model = env.sim.model
    data = env.sim.data
    body_ids = set(_collect_body_subtree_ids(model, int(body_id)))
    geom_bodyid = np.asarray(getattr(model, "geom_bodyid", []), dtype=np.int64).reshape(-1)
    if geom_bodyid.size == 0:
        return float(fallback_radius), float(2.0 * fallback_radius)
    geom_ids = [int(i) for i, bid in enumerate(geom_bodyid) if int(bid) in body_ids]
    if not geom_ids:
        return float(fallback_radius), float(2.0 * fallback_radius)
    geom_group = np.asarray(getattr(model, "geom_group", np.zeros(len(geom_ids))), dtype=np.int64).reshape(-1)
    collision_geom_ids = [gid for gid in geom_ids if geom_group.size <= gid or int(geom_group[gid]) == 0]
    active_geom_ids = collision_geom_ids if collision_geom_ids else geom_ids

    body_pos = np.asarray(data.body_xpos[int(body_id)], dtype=np.float64).reshape(3)
    xy_radii = []
    z_lows = []
    z_highs = []
    for gid in active_geom_ids:
        geom_pos = np.asarray(data.geom_xpos[gid], dtype=np.float64).reshape(3)
        size = np.asarray(model.geom_size[gid], dtype=np.float64).reshape(-1)
        if size.size == 0:
            radius_extent = float(fallback_radius)
            z_extent = float(fallback_radius)
        elif size.size == 1:
            radius_extent = float(size[0])
            z_extent = float(size[0])
        else:
            radius_extent = float(max(size[0], size[1] if size.size > 1 else size[0]))
            z_extent = float(max(size[0], size[1], size[2] if size.size > 2 else size[0]))
        xy_radii.append(float(np.linalg.norm(geom_pos[:2] - body_pos[:2])) + radius_extent)
        z_lows.append(float(geom_pos[2] - z_extent))
        z_highs.append(float(geom_pos[2] + z_extent))

    radius = max(float(fallback_radius), max(xy_radii) if xy_radii else float(fallback_radius))
    height = max(2.0 * float(fallback_radius), (max(z_highs) - min(z_lows)) if z_lows and z_highs else 0.0)
    return float(radius), float(height)


def _get_auto_object_states(env, config: Dict[str, Any]) -> List[AutoObjectState]:
    objects: List[AutoObjectState] = []
    obj_body_id = getattr(env, "obj_body_id", {}) or {}
    objects_dict = getattr(env, "objects_dict", {}) or {}
    fallback_radius = float(config["object_radius_fallback"])
    for name in sorted(objects_dict.keys()):
        if name not in obj_body_id:
            continue
        body_id = int(obj_body_id[name])
        pos = np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).reshape(3).copy()
        quat = np.asarray(env.sim.data.body_xquat[body_id], dtype=np.float64).reshape(4).copy()
        radius, height = _estimate_object_extent(env, body_id, fallback_radius)
        objects.append(
            AutoObjectState(
                name=str(name),
                pos=pos,
                quat_wxyz=quat,
                radius=float(radius),
                height=float(height),
            )
        )
    return objects


def _body_geom_ids(env, body_id: int) -> List[int]:
    model = env.sim.model
    body_ids = set(_collect_body_subtree_ids(model, int(body_id)))
    geom_bodyid = np.asarray(getattr(model, "geom_bodyid", []), dtype=np.int64).reshape(-1)
    return [int(i) for i, bid in enumerate(geom_bodyid) if int(bid) in body_ids]


def _geom_name(model, geom_id: int) -> str:
    try:
        name = model.geom_id2name(int(geom_id))
    except Exception:
        name = None
    return "" if name is None else str(name)


def _build_auto_contact_groups(env) -> Tuple[Dict[int, str], set, set, set]:
    model = env.sim.model
    obj_body_id = getattr(env, "obj_body_id", {}) or {}
    objects_dict = getattr(env, "objects_dict", {}) or {}

    object_geom_to_name: Dict[int, str] = {}
    for name in objects_dict.keys():
        if name not in obj_body_id:
            continue
        for gid in _body_geom_ids(env, int(obj_body_id[name])):
            object_geom_to_name[int(gid)] = str(name)

    gripper_geom_ids = set()
    robot_geom_ids = set()
    table_geom_ids = set()
    for gid in range(int(model.ngeom)):
        name = _geom_name(model, gid).lower()
        if name.startswith("gripper0_") or "finger" in name or "hand" in name:
            gripper_geom_ids.add(int(gid))
            robot_geom_ids.add(int(gid))
        elif name.startswith("robot0_") or "panda" in name:
            robot_geom_ids.add(int(gid))
        if "table" in name or "countertop" in name:
            table_geom_ids.add(int(gid))
    return object_geom_to_name, gripper_geom_ids, robot_geom_ids, table_geom_ids


def _start_auto_episode_metrics(env, config: Dict[str, Any]) -> AutoEpisodeMetrics:
    objects = _get_auto_object_states(env, config)
    return AutoEpisodeMetrics(
        initial_pos={obj.name: obj.pos.copy() for obj in objects},
        initial_quat={obj.name: _normalize_quat_wxyz(obj.quat_wxyz).astype(np.float64) for obj in objects},
        max_z_lift={obj.name: 0.0 for obj in objects},
    )


def _update_auto_episode_metrics(
    env,
    metrics: AutoEpisodeMetrics,
    config: Dict[str, Any],
    contact_groups: Tuple[Dict[int, str], set, set, set],
) -> None:
    object_geom_to_name, gripper_geom_ids, robot_geom_ids, table_geom_ids = contact_groups
    data = env.sim.data
    for idx in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[idx]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        o1, o2 = object_geom_to_name.get(g1), object_geom_to_name.get(g2)
        if (g1 in gripper_geom_ids and o2 is not None) or (g2 in gripper_geom_ids and o1 is not None):
            metrics.eef_object_contacts += 1
        if o1 is not None and o2 is not None and o1 != o2:
            metrics.object_object_contacts += 1
        if (g1 in robot_geom_ids and g2 in table_geom_ids) or (g2 in robot_geom_ids and g1 in table_geom_ids):
            metrics.robot_table_contacts += 1

    lower, upper = _table_xy_bounds(env, extra_margin=float(config["quality_filter"]["leave_table_margin"]))
    table_z = _table_height(env)
    for obj in _get_auto_object_states(env, config):
        if obj.name in metrics.initial_pos:
            z_lift = float(obj.pos[2] - metrics.initial_pos[obj.name][2])
            metrics.max_z_lift[obj.name] = max(float(metrics.max_z_lift.get(obj.name, 0.0)), z_lift)
        if not _in_workspace(obj.xy, lower, upper) or obj.pos[2] < table_z - 0.08:
            metrics.leaves_table = True


def _quat_angle_delta_deg(q0_wxyz: np.ndarray, q1_wxyz: np.ndarray) -> float:
    q0 = _normalize_quat_wxyz(np.asarray(q0_wxyz, dtype=np.float64))
    q1 = _normalize_quat_wxyz(np.asarray(q1_wxyz, dtype=np.float64))
    dot = float(abs(np.dot(q0, q1)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.rad2deg(2.0 * np.arccos(dot)))


def _summarize_auto_episode_metrics(env, metrics: AutoEpisodeMetrics, config: Dict[str, Any]) -> Dict[str, Any]:
    final_objects = {obj.name: obj for obj in _get_auto_object_states(env, config)}
    displacements: Dict[str, float] = {}
    z_deltas: Dict[str, float] = {}
    rotations: Dict[str, float] = {}
    moved_objects = 0
    for name, initial_pos in metrics.initial_pos.items():
        obj = final_objects.get(name)
        if obj is None:
            continue
        disp = float(np.linalg.norm(obj.pos[:2] - initial_pos[:2]))
        z_delta = float(obj.pos[2] - initial_pos[2])
        rot_delta = _quat_angle_delta_deg(metrics.initial_quat[name], obj.quat_wxyz)
        displacements[name] = disp
        z_deltas[name] = z_delta
        rotations[name] = rot_delta
        if (
            disp >= float(config["quality_filter"]["moved_object_displacement"])
            or rot_delta >= float(config["quality_filter"]["moved_object_rotation_deg"])
        ):
            moved_objects += 1

    return {
        "eef_object_contacts": int(metrics.eef_object_contacts),
        "object_object_contacts": int(metrics.object_object_contacts),
        "robot_table_contacts": int(metrics.robot_table_contacts),
        "leaves_table": bool(metrics.leaves_table),
        "max_object_displacement": float(max(displacements.values()) if displacements else 0.0),
        "max_abs_object_z_delta": float(max((abs(v) for v in z_deltas.values()), default=0.0)),
        "max_object_z_lift_during_episode": float(max(metrics.max_z_lift.values()) if metrics.max_z_lift else 0.0),
        "max_object_rotation_deg": float(max(rotations.values()) if rotations else 0.0),
        "moved_objects": int(moved_objects),
        "object_displacements": displacements,
        "object_z_deltas": z_deltas,
        "object_rotation_deg": rotations,
    }


def _evaluate_auto_episode_quality(
    env,
    metrics: AutoEpisodeMetrics,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[bool, str, Dict[str, Any]]:
    summary = _summarize_auto_episode_metrics(env, metrics, config)
    qcfg = config["quality_filter"]
    if not bool(qcfg.get("enabled", True)):
        summary["decision_reason"] = "filter_disabled"
        return True, "unfiltered", summary

    if summary["leaves_table"]:
        summary["decision_reason"] = "object_left_table"
        return False, "reject_left_table", summary
    if summary["max_abs_object_z_delta"] > float(qcfg["max_object_z_delta"]):
        summary["decision_reason"] = "object_z_delta_too_large"
        return False, "reject_lifted_object", summary
    if summary["max_object_z_lift_during_episode"] > float(qcfg["max_object_z_lift_during_episode"]):
        summary["decision_reason"] = "object_lifted_during_episode"
        return False, "reject_lifted_object", summary
    if summary["robot_table_contacts"] > int(qcfg["max_robot_table_contacts"]):
        summary["decision_reason"] = "robot_table_collision"
        return False, "reject_robot_table", summary

    has_eef_contact = int(summary["eef_object_contacts"]) >= int(qcfg["min_eef_object_contacts"])
    has_any_eef_contact = int(summary["eef_object_contacts"]) > 0
    has_object_contact = int(summary["object_object_contacts"]) > 0
    max_disp = float(summary["max_object_displacement"])
    max_rot = float(summary["max_object_rotation_deg"])
    has_keep_motion = (
        max_disp >= float(qcfg["min_keep_displacement"])
        or max_rot >= float(qcfg["min_keep_rotation_deg"])
        or int(summary["moved_objects"]) > 0
    )

    if not has_any_eef_contact and not has_object_contact:
        keep = float(rng.uniform()) < float(qcfg["no_contact_keep_probability"])
        summary["decision_reason"] = "no_contact_sampled" if keep else "no_contact_dropped"
        return keep, "no_contact", summary
    if has_object_contact:
        if has_keep_motion:
            summary["decision_reason"] = "object_object_contact"
            return True, "object_object", summary
        keep = float(rng.uniform()) < float(qcfg["object_object_small_motion_keep_probability"])
        summary["decision_reason"] = "object_object_small_motion_sampled" if keep else "object_object_small_motion_dropped"
        return keep, "object_object_small_motion", summary
    if max_disp >= float(qcfg["significant_displacement"]) or max_rot >= float(qcfg["significant_rotation_deg"]):
        summary["decision_reason"] = "significant_motion"
        return True, "significant_motion", summary
    if has_eef_contact and has_keep_motion:
        summary["decision_reason"] = "eef_contact_motion"
        return True, "eef_contact_motion", summary

    keep = float(rng.uniform()) < float(qcfg["weak_contact_keep_probability"])
    summary["decision_reason"] = "weak_contact_sampled" if keep else "weak_contact_dropped"
    return keep, "weak_contact", summary


def _valid_auto_objects(env, config: Dict[str, Any]) -> List[AutoObjectState]:
    table_z = _table_height(env)
    lower, upper = _workspace_bounds(env, config)
    valid = []
    for obj in _get_auto_object_states(env, config):
        if not _in_workspace(obj.xy, lower, upper):
            continue
        if obj.pos[2] < table_z - 0.10 or obj.pos[2] > table_z + 0.25:
            continue
        valid.append(obj)
    return valid


def _softmax_sample(values: np.ndarray, rng: np.random.Generator, temperature: float = 1.0) -> int:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return 0
    scaled = values / max(float(temperature), 1e-8)
    scaled -= float(np.max(scaled))
    probs = np.exp(scaled)
    probs /= max(float(probs.sum()), 1e-12)
    return int(rng.choice(values.size, p=probs))


def _select_target_object(
    valid_objects: Sequence[AutoObjectState],
    push_counts: Dict[str, int],
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> AutoObjectState:
    if not valid_objects:
        raise RuntimeError("No valid objects available for automatic pushing.")
    mode = _weighted_choice(config["target_sampling_weights"], rng)
    if mode == "nearest_neighbor" and len(valid_objects) > 1:
        nearest_dists = []
        for obj in valid_objects:
            dists = [float(np.linalg.norm(obj.xy - other.xy)) for other in valid_objects if other.name != obj.name]
            nearest_dists.append(min(dists) if dists else 1.0)
        idx = _softmax_sample(-np.asarray(nearest_dists), rng, temperature=0.05)
        return valid_objects[idx]
    if mode == "least_pushed":
        scores = -np.asarray([float(push_counts.get(obj.name, 0)) for obj in valid_objects], dtype=np.float64)
        idx = _softmax_sample(scores, rng, temperature=1.0)
        return valid_objects[idx]
    return valid_objects[int(rng.integers(0, len(valid_objects)))]


def _sample_push_direction(
    target: AutoObjectState,
    valid_objects: Sequence[AutoObjectState],
    push_type: str,
    workspace_center: np.ndarray,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    others = [obj for obj in valid_objects if obj.name != target.name]
    direction: Optional[np.ndarray]
    if push_type == "object_to_object" and others:
        dists = np.asarray([np.linalg.norm(obj.xy - target.xy) for obj in others], dtype=np.float64)
        receiver = others[int(np.argmin(dists))]
        direction = _unit_vector(receiver.xy - target.xy)
        noise = _sample_uniform_range([-config["angle_noise_degrees"]["object_to_object"], config["angle_noise_degrees"]["object_to_object"]], rng)
        direction = None if direction is None else _unit_vector(_rotate_xy(direction, noise))
    elif push_type == "cluster" and others:
        cluster_center = np.mean(np.asarray([obj.xy for obj in others], dtype=np.float64), axis=0)
        direction = _unit_vector(cluster_center - target.xy)
        noise = _sample_uniform_range([-config["angle_noise_degrees"]["cluster"], config["angle_noise_degrees"]["cluster"]], rng)
        direction = None if direction is None else _unit_vector(_rotate_xy(direction, noise))
    elif push_type == "boundary_recovery":
        outward = _unit_vector(target.xy - np.asarray(workspace_center, dtype=np.float64).reshape(2))
        if outward is None:
            return _random_unit_vector(rng)
        num_candidates = max(1, int(config["edge_direction_candidates"]))
        candidate_angles = np.linspace(-90.0, 90.0, num_candidates)
        base_angle = float(candidate_angles[int(rng.integers(0, len(candidate_angles)))])
        noise = _sample_uniform_range(
            [-config["edge_direction_noise_degrees"], config["edge_direction_noise_degrees"]],
            rng,
        )
        direction = _unit_vector(_rotate_xy(outward, base_angle + noise))
    else:
        direction = _random_unit_vector(rng)
    return direction


def _sample_z_push_offset(target: AutoObjectState, config: Dict[str, Any], rng: np.random.Generator) -> float:
    zcfg = config.get("adaptive_z_push", {})
    if not bool(zcfg.get("enabled", False)):
        return _sample_uniform_range(config["z_push_range"], rng)

    height = max(float(target.height), 1e-6)
    radius = max(float(target.radius), float(config["object_radius_fallback"]))
    h_frac = _sample_uniform_range(zcfg["height_fraction_range"], rng)
    r_frac = _sample_uniform_range(zcfg["radius_fraction_range"], rng)
    height_offset = h_frac * height
    radius_offset = r_frac * radius
    offset = min(height_offset, radius_offset)
    return float(np.clip(offset, float(zcfg["min_offset"]), float(zcfg["max_offset"])))


def _compute_auto_push_params(
    env,
    valid_objects: Sequence[AutoObjectState],
    target: AutoObjectState,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> Optional[AutoPushParams]:
    lower, upper = _workspace_bounds(env, config)
    center = _workspace_center(env)
    half_diag = float(np.linalg.norm(0.5 * _table_full_size(env)))
    dist_from_center = float(np.linalg.norm(target.xy - center))
    boundary_threshold = float(config["boundary_threshold_ratio"]) * max(half_diag, 1e-6)
    push_type = "boundary_recovery" if dist_from_center > boundary_threshold else _weighted_choice(config["push_type_weights"], rng)
    if push_type in {"object_to_object", "cluster"} and len(valid_objects) <= 1:
        push_type = "random_free"

    direction = _sample_push_direction(target, valid_objects, push_type, center, config, rng)
    if direction is None:
        return None
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    obj_radius = max(float(target.radius), float(config["object_radius_fallback"]))
    planning_radius = obj_radius + float(config["planning_radius_padding"])
    if push_type == "grazing":
        lateral_abs = _sample_uniform_range(config["lateral_offset_ranges"]["grazing_abs"], rng) * planning_radius
        lateral_offset = float(lateral_abs * rng.choice([-1.0, 1.0]))
    elif push_type == "near_miss_or_weak":
        lateral_abs = _sample_uniform_range(config["lateral_offset_ranges"]["near_miss_abs"], rng) * planning_radius
        lateral_offset = float(lateral_abs * rng.choice([-1.0, 1.0]))
    else:
        lo_hi = config["lateral_offset_ranges"]["default"]
        lateral_offset = _sample_uniform_range([float(lo_hi[0]) * planning_radius, float(lo_hi[1]) * planning_radius], rng)

    length_key = "weak" if push_type == "near_miss_or_weak" and float(rng.uniform()) < 0.5 else "default"
    push_length = _sample_uniform_range(config["push_length_ranges"][length_key], rng)
    start_xy = target.xy - (planning_radius + float(config["ee_radius"]) + float(config["contact_margin"])) * direction
    start_xy = start_xy + lateral_offset * normal
    approach_xy = start_xy - float(config["descent_extra_clearance"]) * direction
    end_xy = start_xy + push_length * direction
    if (
        not _in_workspace(approach_xy, lower, upper)
        or not _in_workspace(start_xy, lower, upper)
        or not _in_workspace(end_xy, lower, upper)
    ):
        return None

    table_z = _table_height(env)
    z_push = table_z + _sample_z_push_offset(target, config, rng)
    z_limits = config["z_push_limits"]
    z_push = float(np.clip(z_push, table_z + float(z_limits[0]), table_z + float(z_limits[1])))
    gripper_mode = _weighted_choice(config["gripper_weights"], rng)
    gripper_cmd = float(config["gripper_commands"].get(gripper_mode, 0.0))
    return AutoPushParams(
        target_name=target.name,
        push_type=push_type,
        approach_xy=approach_xy.astype(np.float64),
        start_xy=start_xy.astype(np.float64),
        end_xy=end_xy.astype(np.float64),
        z_push=z_push,
        gripper_cmd=gripper_cmd,
    )


def _auto_delta_action_to_target(
    env,
    target_pos: np.ndarray,
    gripper_cmd: float,
    config: Dict[str, Any],
    controller_cfg: dict,
    speed_scale: float,
) -> Tuple[np.ndarray, float]:
    current_pose = _snapshot_controller_ee_pose(env)
    target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3)
    pos_scale, _rot_scale = _controller_output_scales(controller_cfg)
    delta_pos = target_pos - current_pose[:3]
    distance = float(np.linalg.norm(delta_pos))
    kp = float(config["osc_kp"]) * float(speed_scale)
    max_delta = float(config["osc_max_delta"]) * float(speed_scale)
    delta_pos_cmd_m = np.clip(kp * delta_pos, -max_delta, max_delta)
    delta_pos_cmd = np.clip(delta_pos_cmd_m / pos_scale, -1.0, 1.0).astype(np.float32, copy=False)
    action_dim = int(getattr(env, "action_dim", 0))
    if action_dim == 7:
        action = np.zeros(7, dtype=np.float32)
        action[:3] = delta_pos_cmd
        action[3:6] = 0.0
        action[6] = float(np.clip(gripper_cmd, -1.0, 1.0))
        return action, distance
    if action_dim == 6:
        action = np.zeros(6, dtype=np.float32)
        action[:3] = delta_pos_cmd
        action[3:6] = 0.0
        return action, distance
    raise RuntimeError(f"Unsupported env.action_dim={action_dim}; expected 6 or 7.")


def _auto_move_ee_to(
    env,
    target_pos: np.ndarray,
    gripper_cmd: float,
    buffer: EpisodeBuffer,
    config: Dict[str, Any],
    controller_cfg: dict,
    speed_scale: float,
    metrics: AutoEpisodeMetrics,
    contact_groups: Tuple[Dict[int, str], set, set, set],
    max_total_steps: int,
    max_steps: Optional[int] = None,
    render: bool = False,
) -> bool:
    step_limit = int(max_steps if max_steps is not None else config["max_steps_per_waypoint"])
    for _ in range(step_limit):
        if buffer.num_steps >= int(max_total_steps):
            return False
        action, distance = _auto_delta_action_to_target(
            env,
            target_pos,
            gripper_cmd,
            config,
            controller_cfg,
            speed_scale,
        )
        if distance <= float(config["pos_tolerance"]):
            return True
        _record_step(env, action, buffer)
        _update_auto_episode_metrics(env, metrics, config, contact_groups)
        if render:
            env.render()
    return False


def _auto_record_settle_steps(
    env,
    gripper_cmd: float,
    buffer: EpisodeBuffer,
    metrics: AutoEpisodeMetrics,
    contact_groups: Tuple[Dict[int, str], set, set, set],
    config: Dict[str, Any],
    max_total_steps: int,
    settle_steps: int,
    render: bool = False,
) -> None:
    action_dim = int(getattr(env, "action_dim", 0))
    action = np.zeros((action_dim,), dtype=np.float32)
    if action_dim == 7:
        action[-1] = float(np.clip(gripper_cmd, -1.0, 1.0))
    for _ in range(max(0, int(settle_steps))):
        if buffer.num_steps >= int(max_total_steps):
            return
        _record_step(env, action, buffer)
        _update_auto_episode_metrics(env, metrics, config, contact_groups)
        if render:
            env.render()


def _execute_auto_push(
    env,
    params: AutoPushParams,
    buffer: EpisodeBuffer,
    config: Dict[str, Any],
    controller_cfg: dict,
    speed_scale: float,
    metrics: AutoEpisodeMetrics,
    contact_groups: Tuple[Dict[int, str], set, set, set],
    max_total_steps: int,
    render: bool = False,
) -> None:
    table_z = _table_height(env)
    clear_z = table_z + float(config["clearance_height"])
    current_pose = _snapshot_controller_ee_pose(env)
    current_clear = np.asarray([current_pose[0], current_pose[1], clear_z], dtype=np.float32)
    approach_clear = np.asarray([params.approach_xy[0], params.approach_xy[1], clear_z], dtype=np.float32)
    approach_push = np.asarray([params.approach_xy[0], params.approach_xy[1], params.z_push], dtype=np.float32)
    start_push = np.asarray([params.start_xy[0], params.start_xy[1], params.z_push], dtype=np.float32)
    end_push = np.asarray([params.end_xy[0], params.end_xy[1], params.z_push], dtype=np.float32)
    end_clear = np.asarray([params.end_xy[0], params.end_xy[1], clear_z], dtype=np.float32)

    if not _auto_move_ee_to(
        env,
        current_clear,
        params.gripper_cmd,
        buffer,
        config,
        controller_cfg,
        speed_scale,
        metrics,
        contact_groups,
        max_total_steps,
        render=render,
    ):
        return
    if not _auto_move_ee_to(
        env,
        approach_clear,
        params.gripper_cmd,
        buffer,
        config,
        controller_cfg,
        speed_scale,
        metrics,
        contact_groups,
        max_total_steps,
        render=render,
    ):
        return
    if not _auto_move_ee_to(
        env,
        approach_push,
        params.gripper_cmd,
        buffer,
        config,
        controller_cfg,
        speed_scale,
        metrics,
        contact_groups,
        max_total_steps,
        render=render,
    ):
        return
    if not _auto_move_ee_to(
        env,
        start_push,
        params.gripper_cmd,
        buffer,
        config,
        controller_cfg,
        speed_scale,
        metrics,
        contact_groups,
        max_total_steps,
        render=render,
    ):
        return
    if buffer.num_steps >= int(max_total_steps):
        return

    for alpha in np.linspace(0.0, 1.0, int(config["push_waypoints"])):
        target_pos = ((1.0 - float(alpha)) * start_push + float(alpha) * end_push).astype(np.float32, copy=False)
        if not _auto_move_ee_to(
            env,
            target_pos,
            params.gripper_cmd,
            buffer,
            config,
            controller_cfg,
            speed_scale,
            metrics,
            contact_groups,
            max_total_steps,
            max_steps=int(config["push_steps_per_waypoint"]),
            render=render,
        ):
            return
        if buffer.num_steps >= int(max_total_steps):
            return
    if not _auto_move_ee_to(
        env,
        end_clear,
        params.gripper_cmd,
        buffer,
        config,
        controller_cfg,
        speed_scale,
        metrics,
        contact_groups,
        max_total_steps,
        render=render,
    ):
        return
    _auto_record_settle_steps(
        env,
        params.gripper_cmd,
        buffer,
        metrics,
        contact_groups,
        config,
        max_total_steps=max_total_steps,
        settle_steps=int(config["settle_steps"]),
        render=render,
    )


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
    episode_attrs: Optional[Dict[str, Any]] = None,
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
        if episode_attrs:
            for key, value in episode_attrs.items():
                if isinstance(value, (dict, list, tuple)):
                    demo_group.attrs[str(key)] = json.dumps(_jsonable(value))
                else:
                    demo_group.attrs[str(key)] = _jsonable(value)

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


def _restore_collection_scene(
    env,
    is_hdf5: bool,
    source_path: Path,
    demo_keys: Sequence[str],
    rng: np.random.Generator,
) -> EpisodeSeed:
    if is_hdf5:
        demo_key = str(rng.choice(demo_keys))
        seed = _load_episode_seed(source_path, demo_key)
        _reset_to_episode_seed(env, seed)
        print(f"[RESET] restored {demo_key}")
        return seed
    seed = _reset_from_bddl(env)
    print("[RESET] restored BDDL scene")
    return seed


def _run_auto_collection(
    env,
    args: argparse.Namespace,
    output_path: Path,
    source_path: Path,
    is_hdf5: bool,
    demo_keys: Sequence[str],
    rng: np.random.Generator,
    env_kwargs: dict,
) -> None:
    config = _load_auto_config(args.auto_config)
    target_episodes = int(args.auto_collect)
    if target_episodes <= 0:
        return
    if int(args.max_steps) <= 0:
        raise RuntimeError("--max-steps must be positive for --auto-collect.")

    if not args.no_render:
        _install_dual_camera_viewer(env, args)

    controller_cfg = dict(env_kwargs.get("controller_configs", {}) or {})
    saved = 0
    discarded = 0
    last_len = 0
    push_counts: Dict[str, int] = {}
    empty_resets = 0

    print(
        f"[AUTO] collecting {target_episodes} episodes "
        f"pushes_per_episode={config['pushes_per_episode']} max_steps={int(args.max_steps)}"
    )
    while saved < target_episodes:
        _restore_collection_scene(env, is_hdf5, source_path, demo_keys, rng)
        if not args.no_render:
            _install_dual_camera_viewer(env, args)
            env.render()
        buffer = EpisodeBuffer.start(env)
        metrics = _start_auto_episode_metrics(env, config)
        contact_groups = _build_auto_contact_groups(env)
        episode_speed_scale = _sample_uniform_range(config["episode_speed_scale_range"], rng)
        primitive_idx = 0
        attempts = 0
        while primitive_idx < int(config["pushes_per_episode"]) and buffer.num_steps < int(args.max_steps):
            if attempts >= int(config["max_sampling_attempts"]):
                print(f"[AUTO] sampling attempts exhausted after {attempts} tries")
                break
            valid_objects = _valid_auto_objects(env, config)
            if not valid_objects:
                print("[AUTO] no valid movable objects in workspace")
                break
            target = _select_target_object(valid_objects, push_counts, config, rng)
            params = _compute_auto_push_params(env, valid_objects, target, config, rng)
            attempts += 1
            if params is None:
                continue

            push_counts[params.target_name] = int(push_counts.get(params.target_name, 0)) + 1
            before_steps = buffer.num_steps
            print(
                f"[AUTO] push={primitive_idx + 1}/{int(config['pushes_per_episode'])} "
                f"type={params.push_type} target={params.target_name} "
                f"speed={episode_speed_scale:.2f} steps={buffer.num_steps}"
            )
            _execute_auto_push(
                env,
                params,
                buffer,
                config,
                controller_cfg,
                episode_speed_scale,
                metrics,
                contact_groups,
                max_total_steps=int(args.max_steps),
                render=not bool(args.no_render),
            )
            if buffer.num_steps <= before_steps:
                print("[AUTO] primitive produced no steps; stopping this episode")
                break
            primitive_idx += 1

        if buffer.num_steps <= 0:
            discarded += 1
            empty_resets += 1
            if empty_resets >= 10:
                raise RuntimeError("Automatic collection failed to produce any steps after 10 resets.")
            print("[AUTO] empty episode discarded")
            continue

        keep, quality_category, quality_summary = _evaluate_auto_episode_quality(env, metrics, config, rng)
        if not keep:
            discarded += 1
            print(
                f"[AUTO-DISCARD] category={quality_category} reason={quality_summary.get('decision_reason')} "
                f"eef_obj={quality_summary['eef_object_contacts']} obj_obj={quality_summary['object_object_contacts']} "
                f"moved={quality_summary['moved_objects']} max_disp={quality_summary['max_object_displacement']:.4f} "
                f"max_rot={quality_summary['max_object_rotation_deg']:.2f} "
                f"max_z={quality_summary['max_abs_object_z_delta']:.4f} robot_table={quality_summary['robot_table_contacts']}"
            )
            continue

        empty_resets = 0
        episode_attrs = {
            "collection_mode": "auto",
            "auto_quality_category": quality_category,
            "auto_quality_metrics": quality_summary,
            "auto_episode_speed_scale": float(episode_speed_scale),
        }
        demo_key, last_len = _append_episode(output_path, buffer, episode_attrs=episode_attrs)
        saved += 1
        print(
            f"[SAVED] {demo_key} len={last_len} mode=auto quality={quality_category} "
            f"eef_obj={quality_summary['eef_object_contacts']} obj_obj={quality_summary['object_object_contacts']} "
            f"moved={quality_summary['moved_objects']} max_disp={quality_summary['max_object_displacement']:.4f} "
            f"max_rot={quality_summary['max_object_rotation_deg']:.2f}"
        )
        _print_status(
            phase="AUTO",
            saved=saved,
            discarded=discarded,
            current_steps=0,
            last_len=last_len,
            output_path=output_path,
        )


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
    parser = argparse.ArgumentParser(description="Collect LIBERO pushing trajectories with Omega.7 teleoperation or automatic primitives.")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Input .hdf5 or .bddl scene source.")
    parser.add_argument("--output", type=str, default=None, help="Output HDF5 path.")
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
        "--auto-collect",
        type=int,
        default=0,
        help="Collect this many episodes with automatic pushing primitives instead of Omega teleoperation.",
    )
    parser.add_argument(
        "--auto-config",
        type=str,
        default=None,
        help="Optional JSON config for automatic pushing primitive probabilities and controller parameters.",
    )
    parser.add_argument(
        "--dump-auto-config",
        type=str,
        default=None,
        help="Write the default automatic pushing config to this path and exit.",
    )
    parser.add_argument(
        "--dry-run-init",
        action="store_true",
        help="Initialize one scene, then exit without creating output or touching Omega.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dump_auto_config is not None:
        _dump_auto_config(str(args.dump_auto_config))
        return
    if args.output is None:
        raise ValueError("--output is required unless --dump-auto-config is used.")
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
    if int(args.auto_collect) > 0:
        try:
            _run_auto_collection(
                env=env,
                args=args,
                output_path=output_path,
                source_path=source_path,
                is_hdf5=is_hdf5,
                demo_keys=demo_keys,
                rng=rng,
                env_kwargs=env_kwargs,
            )
            return
        finally:
            env.close()

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
