#!/usr/bin/env python3
"""
Record a short RGB + depth (colormap) video from the first connected RealSense device.

Note: On macOS 12+ (Monterey) and newer, librealsense often requires running with
sudo to access USB (see librealsense macOS installation docs).
"""

from __future__ import annotations

import argparse
import os
import pwd
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


def _get_first_device_serial(ctx: rs.context) -> str:
    devs = list(ctx.devices)
    if not devs:
        raise RuntimeError("No RealSense device detected by librealsense.")
    return devs[0].get_info(rs.camera_info.serial_number)


def _start_pipeline(
    *,
    ctx: rs.context,
    serial: str,
    width: int,
    height: int,
    fps_candidates: Tuple[int, ...],
) -> Tuple[rs.pipeline, int]:
    last_err: Optional[BaseException] = None
    for fps in fps_candidates:
        pipeline = rs.pipeline(ctx)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        try:
            pipeline.start(config)

            # On some setups (e.g. marginal USB bandwidth), `start()` can succeed but
            # streaming never actually begins. Verify we can receive at least 1 frame.
            ok = False
            for _ in range(3):
                try:
                    pipeline.wait_for_frames(2000)
                    ok = True
                    break
                except RuntimeError as e:
                    last_err = e
            if not ok:
                raise RuntimeError(f"Frame didn't arrive after starting {width}x{height}@{fps}")
            return pipeline, fps
        except Exception as e:
            last_err = e
            try:
                pipeline.stop()
            except Exception:
                pass
            print(f"[WARN] Failed to start {width}x{height}@{fps}: {type(e).__name__}: {e}")
    raise RuntimeError(f"Failed to start pipeline with candidates={fps_candidates}: {last_err}")


def _maybe_chown_tree(path: Path) -> None:
    """If running under sudo, chown outputs back to the original user for convenience."""
    sudo_user = os.environ.get("SUDO_USER")
    if not sudo_user:
        return

    try:
        pw = pwd.getpwnam(sudo_user)
    except KeyError:
        return

    uid = pw.pw_uid
    gid = pw.pw_gid
    for p in [path, *path.rglob("*")]:
        try:
            os.chown(p, uid, gid)
        except PermissionError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30, help="Preferred FPS; will fall back if needed.")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output directory (default: <repo>/data_collect/realsense_test_<timestamp>).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (repo_root / "data_collect" / f"realsense_test_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = rs.context()
    devs = list(ctx.devices)
    serial = _get_first_device_serial(ctx)
    usb_desc = ""
    try:
        usb_desc = devs[0].get_info(rs.camera_info.usb_type_descriptor)
    except Exception:
        usb_desc = ""
    print(f"Using device serial={serial}" + (f" usb={usb_desc}" if usb_desc else ""))

    # On USB2, starting too aggressive a config can result in `start()` succeeding
    # but no frames arriving. Prefer a conservative fallback order.
    if usb_desc.startswith("2"):
        candidates = [15, 5, args.fps]
    else:
        candidates = [args.fps, 15, 5]
    fps_candidates = tuple(dict.fromkeys(candidates))  # de-dupe while preserving order
    pipeline, fps_used = _start_pipeline(
        ctx=ctx,
        serial=serial,
        width=args.width,
        height=args.height,
        fps_candidates=fps_candidates,
    )
    print(f"Pipeline started at {args.width}x{args.height}@{fps_used}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    color_path = out_dir / "color.mp4"
    depth_path = out_dir / "depth_colormap.mp4"
    color_writer = cv2.VideoWriter(str(color_path), fourcc, float(fps_used), (args.width, args.height))
    depth_writer = cv2.VideoWriter(str(depth_path), fourcc, float(fps_used), (args.width, args.height))
    if not color_writer.isOpened() or not depth_writer.isOpened():
        print("[ERROR] Failed to open video writers. Try installing a full OpenCV build with video codecs.")
        return 4

    align = rs.align(rs.stream.color)
    frame_count = 0
    t0 = time.time()
    try:
        while (time.time() - t0) < float(args.seconds):
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            # Visualize depth as a colormap (8-bit) for mp4 encoding.
            depth_8u = cv2.convertScaleAbs(depth, alpha=0.03)
            depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

            color_writer.write(color)
            depth_writer.write(depth_colormap)
            frame_count += 1
    except KeyboardInterrupt:
        print("Interrupted, stopping early...")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        color_writer.release()
        depth_writer.release()

    _maybe_chown_tree(out_dir)

    duration = time.time() - t0
    print(f"Saved {frame_count} frames in ~{duration:.2f}s")
    print(f"Color video: {color_path}")
    print(f"Depth video:  {depth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
