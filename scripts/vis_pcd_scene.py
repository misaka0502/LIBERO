"""
Visualize scene point cloud trajectories saved by create_pcd_scene.py.

Expected NPZ keys:
  - scene_pcd: (T, N, 3)
  - frame_points: (T,) optional, valid points per frame
  - episode: scalar string optional
  - object_names: (K,) optional

Usage:
  python vis_pcd_scene.py --dataset /path/to/scene_demo5.npz
  python vis_pcd_scene.py --dataset /path/to/scene_demo5.npz --frame 20 --point-size 2.0
"""

import argparse
import sys

import numpy as np
import open3d as o3d


def _to_python_str(value, default=""):
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _to_python_str(value.item(), default=default)
        if value.size == 1:
            return _to_python_str(value.reshape(-1)[0], default=default)
    return str(value)


def load_scene_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    if "scene_pcd" not in data:
        raise KeyError("NPZ missing required key 'scene_pcd'")

    scene_pcd = np.asarray(data["scene_pcd"], dtype=np.float32)
    if scene_pcd.ndim != 3 or scene_pcd.shape[-1] != 3:
        raise ValueError(f"scene_pcd has invalid shape {scene_pcd.shape}, expected (T, N, 3)")

    num_frames = scene_pcd.shape[0]
    padded_points = scene_pcd.shape[1]

    if "frame_points" in data:
        frame_points = np.asarray(data["frame_points"]).reshape(-1).astype(np.int32)
        if frame_points.shape[0] != num_frames:
            raise ValueError(
                f"frame_points length mismatch: {frame_points.shape[0]} vs num_frames={num_frames}"
            )
        frame_points = np.clip(frame_points, 0, padded_points)
    else:
        # Compatibility fallback: infer valid count by dropping all-zero rows.
        valid_mask = ~np.all(np.isclose(scene_pcd, 0.0), axis=2)
        frame_points = valid_mask.sum(axis=1).astype(np.int32)

    episode = _to_python_str(data["episode"], default="") if "episode" in data else ""

    object_names = []
    if "object_names" in data:
        raw = np.asarray(data["object_names"]).reshape(-1)
        object_names = [_to_python_str(x) for x in raw]

    return scene_pcd, frame_points, episode, object_names


def get_frame_points(scene_pcd, frame_points, frame_idx):
    n = int(frame_points[frame_idx])
    return scene_pcd[frame_idx, :n]


def visualize(npz_path, start_frame=0, point_size=3.0):
    scene_pcd, frame_points, episode, object_names = load_scene_npz(npz_path)

    num_frames = scene_pcd.shape[0]
    if num_frames == 0:
        print("No frames to visualize")
        return

    current_frame = int(np.clip(start_frame, 0, num_frames - 1))

    print(f"Episode: {episode if episode else '(unknown)'}")
    print(f"Frames: {num_frames}")
    print(f"Objects (from metadata): {len(object_names)}")
    print(f"Valid points range: [{int(frame_points.min())}, {int(frame_points.max())}]")
    print("Controls: SPACE/D/RIGHT -> next | BACKSPACE/A/LEFT -> prev | ESC/Q -> quit")
    print(f"=== Frame {current_frame}/{num_frames - 1} ===")
    sys.stdout.flush()

    points = get_frame_points(scene_pcd, frame_points, current_frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(np.array([[0.15, 0.75, 0.95]], dtype=np.float32), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Scene Point Cloud", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)

    render_opt = vis.get_render_option()
    render_opt.point_size = float(point_size)

    ctrl = vis.get_view_control()
    ctrl.set_lookat([0.0, 0.0, 0.0])
    ctrl.set_front([0.0, -1.0, 1.0])
    ctrl.set_up([0.0, 0.0, 1.0])
    ctrl.set_zoom(0.8)

    def update_frame(new_idx):
        nonlocal current_frame
        current_frame = int(new_idx % num_frames)

        frame_pts = get_frame_points(scene_pcd, frame_points, current_frame)
        pcd.points = o3d.utility.Vector3dVector(frame_pts)
        frame_colors = np.tile(np.array([[0.15, 0.75, 0.95]], dtype=np.float32), (frame_pts.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(frame_colors)

        vis.update_geometry(pcd)
        print(f"=== Frame {current_frame}/{num_frames - 1} | points={int(frame_points[current_frame])} ===")
        sys.stdout.flush()

    def next_cb(_vis):
        update_frame(current_frame + 1)
        vis.poll_events()
        vis.update_renderer()

    def prev_cb(_vis):
        update_frame(current_frame - 1)
        vis.poll_events()
        vis.update_renderer()

    def quit_cb(_vis):
        vis.close()

    def register_many(keys, callback):
        for k in keys:
            vis.register_key_callback(int(k), callback)

    # Open3D / GLFW keycode compatibility across platforms.
    register_many([32, 68, 262], next_cb)       # SPACE, D, RIGHT
    register_many([8, 259, 65, 263], prev_cb)   # BACKSPACE, BACKSPACE(GLFW), A, LEFT
    register_many([27, 256, 81], quit_cb)       # ESC(ascii), ESC(GLFW), Q

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Visualize scene point cloud trajectories")
    parser.add_argument("--dataset", type=str, required=True, help="Path to scene NPZ file")
    parser.add_argument("--frame", type=int, default=None, help="Start at a specific frame")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--point-size", type=float, default=3.0, help="Open3D point size")

    args = parser.parse_args()
    start = args.frame if args.frame is not None else args.start_frame

    visualize(args.dataset, start_frame=start, point_size=args.point_size)


if __name__ == "__main__":
    main()
