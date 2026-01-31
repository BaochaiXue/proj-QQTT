import os

from qqtt.env import CameraSystem


def main() -> None:
    # Ensure calibrate.pkl is saved next to this script (project root).
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Auto-detect camera count (useful for single-camera setups).
    camera_system: CameraSystem = CameraSystem(WH=[1280, 720], fps=5, num_cam=None)
    camera_system.calibrate()

if __name__ == "__main__":
    main()
