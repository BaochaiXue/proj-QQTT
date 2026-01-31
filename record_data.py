from datetime import datetime
import os
import shutil

from qqtt.env import CameraSystem


def exist_dir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == "__main__":
    # Auto-detect camera count (useful for single-camera setups).
    camera_system: CameraSystem = CameraSystem(num_cam=None)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_collect_dir = os.path.join(base_dir, "data_collect")
    exist_dir(data_collect_dir)
    out_dir = os.path.join(data_collect_dir, current_time)
    camera_system.record(output_path=out_dir)

    # Copy the camera calibration file (if present) to the output path.
    calibrate_path = os.path.join(base_dir, "calibrate.pkl")
    if os.path.isfile(calibrate_path):
        shutil.copy2(calibrate_path, out_dir)
    else:
        print(f"[WARN] calibrate.pkl not found at: {calibrate_path}")
