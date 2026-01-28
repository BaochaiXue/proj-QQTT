from datetime import datetime
from qqtt.env import CameraSystem
import os

def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == "__main__":
    camera_system = CameraSystem()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_collect_dir = os.path.join(base_dir, "data_collect")
    exist_dir(data_collect_dir)
    camera_system.record(
        output_path=os.path.join(data_collect_dir, current_time)
    )
    # Copy the camera calibration file to the output path
    os.system(
        f"cp {os.path.join(base_dir, 'calibrate.pkl')} {os.path.join(data_collect_dir, current_time)}"
    )
