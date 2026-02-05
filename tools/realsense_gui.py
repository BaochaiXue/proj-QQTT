#!/usr/bin/env python3
"""
Simple GUI viewer for up to 3 RealSense cameras.

Shows color (top) + depth colormap (bottom) per camera, tiled in a grid.
On macOS 12+ you may need to run with sudo for USB access.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


def _start_pipeline(
    *,
    ctx: rs.context,
    serial: str,
    profiles: Tuple[Tuple[int, int, int], ...],
) -> Tuple[rs.pipeline, Tuple[int, int, int]]:
    last_err: Optional[BaseException] = None
    for width, height, fps in profiles:
        pipeline = rs.pipeline(ctx)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        try:
            pipeline.start(config)
            # Make sure frames actually arrive.
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
            return pipeline, (width, height, fps)
        except Exception as e:
            last_err = e
            try:
                pipeline.stop()
            except Exception:
                pass
            print(f"[WARN] Failed to start {width}x{height}@{fps}: {type(e).__name__}: {e}")
            # Give libusb a brief chance to release interfaces before retry.
            time.sleep(0.2)
    raise RuntimeError(f"Failed to start pipeline with candidates={profiles}: {last_err}")


def _build_profiles(
    *,
    req_width: int,
    req_height: int,
    req_fps: int,
    usb_desc: str,
    total_cams: int,
) -> Tuple[Tuple[int, int, int], ...]:
    profiles: List[Tuple[int, int, int]] = []

    if total_cams >= 3:
        # In 3-camera mode under WSL/usbip, start conservatively first.
        profiles.extend(
            [
                (848, 480, 15),
                (640, 480, 15),
                (640, 480, 5),
                (req_width, req_height, 15),
                (req_width, req_height, 5),
                (848, 480, 30),
                (req_width, req_height, req_fps),
            ]
        )
    else:
        # Try requested profile first (or safer fps for USB2).
        if usb_desc.startswith("2"):
            profiles.append((req_width, req_height, min(req_fps, 15)))
        else:
            profiles.append((req_width, req_height, req_fps))
        profiles.extend(
            [
                (848, 480, 15),
                (848, 480, 5),
                (640, 480, 30),
                (640, 480, 15),
                (640, 480, 5),
            ]
        )

    # De-duplicate while preserving order.
    seen = set()
    uniq: List[Tuple[int, int, int]] = []
    for p in profiles:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return tuple(uniq)


def _safe_wait_frames(pipeline: rs.pipeline, timeout_ms: int = 100) -> Optional[rs.frameset]:
    try:
        return pipeline.wait_for_frames(timeout_ms)
    except RuntimeError:
        return None


def _make_panel(
    *,
    color: np.ndarray,
    depth: np.ndarray,
    label: str,
) -> np.ndarray:
    depth_8u = cv2.convertScaleAbs(depth, alpha=0.03)
    depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    panel = np.vstack([color, depth_colormap])
    cv2.putText(
        panel,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _empty_panel(width: int, height: int, label: str) -> np.ndarray:
    panel = np.zeros((height * 2, width, 3), dtype=np.uint8)
    cv2.putText(
        panel,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _tile_panels(panels: List[np.ndarray], panel_h: int, panel_w: int) -> np.ndarray:
    if not panels:
        return np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    cols = 2
    rows = int(math.ceil(len(panels) / cols))
    grid = np.zeros((rows * panel_h, cols * panel_w, 3), dtype=np.uint8)
    for idx, panel in enumerate(panels):
        r = idx // cols
        c = idx % cols
        y0, y1 = r * panel_h, (r + 1) * panel_h
        x0, x1 = c * panel_w, (c + 1) * panel_w
        grid[y0:y1, x0:x1] = panel
    return grid


def main() -> int:
    parser = argparse.ArgumentParser()
    # Match cameras_calibrate.py defaults (1280x720).
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-cams", type=int, default=3)
    args = parser.parse_args()

    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        print("No RealSense device detected by librealsense.")
        return 2

    devices = devices[: args.max_cams]
    serials = []
    for dev in devices:
        try:
            serials.append(dev.get_info(rs.camera_info.serial_number))
        except Exception:
            serials.append("unknown")
    print(f"Detected {len(devices)} camera(s): {', '.join(serials)}")

    cams: List[Dict[str, object]] = []
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        usb_desc = ""
        try:
            usb_desc = dev.get_info(rs.camera_info.usb_type_descriptor)
        except Exception:
            usb_desc = ""

        profiles = _build_profiles(
            req_width=args.width,
            req_height=args.height,
            req_fps=args.fps,
            usb_desc=usb_desc,
            total_cams=len(devices),
        )
        try:
            pipeline, (stream_w, stream_h, fps_used) = _start_pipeline(
                ctx=ctx,
                serial=serial,
                profiles=profiles,
            )
        except Exception as e:
            print(f"[ERROR] Could not start {serial}: {type(e).__name__}: {e}")
            continue

        cams.append(
            {
                "serial": serial,
                "usb": usb_desc,
                "pipeline": pipeline,
                "align": rs.align(rs.stream.color),
                "fps": fps_used,
                "stream_w": stream_w,
                "stream_h": stream_h,
                "last_panel": _empty_panel(args.width, args.height, f"{serial} (waiting)"),
            }
        )
        print(
            f"Started {serial} usb={usb_desc or 'unknown'} "
            f"at {stream_w}x{stream_h}@{fps_used}"
        )

    if not cams:
        print("Failed to start any camera.")
        return 3
    print(f"Running with {len(cams)} camera(s).")

    panel_h = args.height * 2
    panel_w = args.width

    try:
        while True:
            panels: List[np.ndarray] = []
            for cam in cams:
                pipeline = cam["pipeline"]
                frames = _safe_wait_frames(pipeline, timeout_ms=100)
                if frames:
                    frames = cam["align"].process(frames)
                    depth = frames.get_depth_frame()
                    color = frames.get_color_frame()
                    if depth and color:
                        color_np = np.asanyarray(color.get_data())
                        depth_np = np.asanyarray(depth.get_data())
                        if color_np.shape[1] != args.width or color_np.shape[0] != args.height:
                            color_np = cv2.resize(
                                color_np, (args.width, args.height), interpolation=cv2.INTER_LINEAR
                            )
                        if depth_np.shape[1] != args.width or depth_np.shape[0] != args.height:
                            depth_np = cv2.resize(
                                depth_np, (args.width, args.height), interpolation=cv2.INTER_NEAREST
                            )
                        label = (
                            f"{cam['serial']} usb={cam['usb']} "
                            f"{cam['stream_w']}x{cam['stream_h']}@{cam['fps']}fps"
                        )
                        cam["last_panel"] = _make_panel(
                            color=color_np, depth=depth_np, label=label
                        )
                panels.append(cam["last_panel"])

            grid = _tile_panels(panels, panel_h, panel_w)
            cv2.imshow("RealSense Viewer", grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        for cam in cams:
            try:
                cam["pipeline"].stop()
            except Exception:
                pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
