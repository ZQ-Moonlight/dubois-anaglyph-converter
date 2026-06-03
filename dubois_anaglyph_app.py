import argparse
import glob
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import cv2
import numpy as np


DUBOIS_MATRIX = np.array(
    [
        [0.4561, 0.5005, 0.1764, -0.0435, -0.0879, -0.0016],
        [-0.0401, -0.0378, -0.0158, 0.3785, 0.7336, -0.0185],
        [-0.0152, -0.0206, -0.0055, -0.0722, -0.1130, 1.2264],
    ],
    dtype=np.float32,
)

VIDEO_SEPARATE = "separate"
VIDEO_FULL_SBS = "full_sbs"
VIDEO_HALF_SBS = "half_sbs"

PHOTO_SEPARATE = "photo_separate"
PHOTO_FULL_SBS = "photo_full_sbs"
PHOTO_HALF_SBS = "photo_half_sbs"

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output"

_TORCH_MODULE = None
_TORCH_CHECKED = False
_HARDWARE_CACHE = None

PAPER_REFERENCES = [
    {
        "title": "A projection method to generate anaglyph stereo images",
        "authors": "Eric Dubois",
        "year": "2001",
        "url": "https://doi.org/10.1109/ICASSP.2001.941256",
        "note": "Dubois 矩阵的核心来源：用投影/最小二乘思路降低串扰和颜色竞争。",
    },
    {
        "title": "Conversion of a Stereo Pair to Anaglyph with the Least-Squares Projection Method",
        "authors": "Eric Dubois",
        "year": "2009",
        "url": "https://www.site.uottawa.ca/~edubois/anaglyph/",
        "note": "进一步讨论立体图像到 anaglyph 的颜色变换和显示适配。",
    },
    {
        "title": "Visual comfort of binocular and 3D displays",
        "authors": "Frank L. Kooi; Alexander Toet",
        "year": "2004",
        "url": "https://doi.org/10.1016/j.displa.2004.07.004",
        "note": "视觉舒适度相关：视差、错位、串扰和亮度/对比度都会影响观看压力。",
    },
    {
        "title": "Visual discomfort and visual fatigue of stereoscopic displays",
        "authors": "Lambooij et al.",
        "year": "2009",
        "url": "https://doi.org/10.2352/J.ImagingSci.Technol.2009.53.3.030201",
        "note": "综述立体显示疲劳因素，支持增加垂直校正、对比度和饱和度调节。",
    },
    {
        "title": "Characterization of crosstalk in stereoscopic display devices",
        "authors": "Zafar; Badano",
        "year": "2015",
        "url": "https://doi.org/10.1002/jsid.279",
        "note": "串扰会降低立体画质并增加不适感，适合指导鬼影抑制和通道增益调节。",
    },
]


def default_output_path(name):
    return str(OUTPUT_DIR / name)


def normalize_output_path(path):
    if not path:
        raise ValueError("请选择输出路径。")
    output = Path(path)
    if not output.is_absolute():
        output = APP_DIR / output
    return output


def clamp_float(value, default, minimum=None, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if minimum is not None:
        number = max(float(minimum), number)
    if maximum is not None:
        number = min(float(maximum), number)
    return number


def bool_setting(settings, name, default=False):
    value = settings.get(name, default)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def format_duration(seconds):
    seconds = max(0.0, float(seconds or 0.0))
    whole = int(seconds + 0.5)
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def find_executable(name):
    found = shutil.which(name)
    if found:
        return found
    candidates = [
        APP_DIR / "ffmpeg" / "bin" / f"{name}.exe",
        APP_DIR / "bin" / f"{name}.exe",
        Path("C:/ffmpeg/bin") / f"{name}.exe",
        Path("C:/Program Files/ffmpeg/bin") / f"{name}.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    for folder in APP_DIR.glob("ffmpeg-*"):
        candidate = folder / "bin" / f"{name}.exe"
        if candidate.exists():
            return str(candidate)
    return None


def get_torch_module():
    global _TORCH_CHECKED, _TORCH_MODULE
    if not _TORCH_CHECKED:
        _TORCH_CHECKED = True
        try:
            import torch

            _TORCH_MODULE = torch
        except Exception:
            _TORCH_MODULE = None
    return _TORCH_MODULE


def detect_hardware(force=False):
    global _HARDWARE_CACHE
    if _HARDWARE_CACHE is not None and not force:
        return _HARDWARE_CACHE

    ffmpeg_path = find_executable("ffmpeg")
    ffprobe_path = find_executable("ffprobe")
    cv_cuda_devices = 0
    try:
        if hasattr(cv2, "cuda"):
            cv_cuda_devices = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        cv_cuda_devices = 0

    torch = get_torch_module()
    torch_cuda = False
    torch_device = ""
    torch_version = ""
    if torch is not None:
        try:
            torch_version = str(torch.__version__)
            torch_cuda = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
            if torch_cuda:
                torch_device = torch.cuda.get_device_name(0)
        except Exception:
            torch_cuda = False

    backend = "torch_cuda" if torch_cuda else "cpu"
    _HARDWARE_CACHE = {
        "ok": True,
        "backend": backend,
        "backend_label": "Torch CUDA" if torch_cuda else "CPU / NumPy",
        "torch_available": torch is not None,
        "torch_version": torch_version,
        "torch_cuda": torch_cuda,
        "torch_device": torch_device,
        "opencv_cuda_devices": cv_cuda_devices,
        "ffmpeg": ffmpeg_path,
        "ffprobe": ffprobe_path,
        "audio_mux": bool(ffmpeg_path),
    }
    return _HARDWARE_CACHE


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"无法编码输出图片：{path}")
    encoded.tofile(str(path))


def resize_to_match(left, right):
    h, w = left.shape[:2]
    if right.shape[:2] != (h, w):
        right = cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA)
    return left, right


def shift_image(image, dx=0.0, dy=0.0):
    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return image
    h, w = image.shape[:2]
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def shift_horizontal(image, dx):
    return shift_image(image, dx=dx, dy=0.0)


def trim_and_restore_width(image, convergence_px):
    trim = int(math.ceil(abs(convergence_px)))
    if trim <= 0:
        return image
    h, w = image.shape[:2]
    trim = min(trim, max(0, (w - 16) // 4))
    if trim <= 0 or w - trim * 2 < 16:
        return image
    cropped = image[:, trim : w - trim]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)


def srgb_to_linear_np(values):
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb_np(values):
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * np.power(values, 1.0 / 2.4) - 0.055)


def preprocess_rgb_np(rgb, gain, saturation, contrast, brightness):
    rgb = rgb * gain
    if abs(saturation - 1.0) > 0.001:
        gray = (
            rgb[..., 0:1] * 0.2126
            + rgb[..., 1:2] * 0.7152
            + rgb[..., 2:3] * 0.0722
        )
        rgb = gray + (rgb - gray) * saturation
    if abs(contrast - 1.0) > 0.001:
        rgb = (rgb - 0.5) * contrast + 0.5
    if abs(brightness) > 0.001:
        rgb = rgb + brightness
    return np.clip(rgb, 0.0, 1.0)


def advanced_values(
    convergence_px=0.0,
    trim_edges=True,
    vertical_offset_px=0.0,
    swap_eyes=False,
    left_gain=100.0,
    right_gain=100.0,
    saturation=100.0,
    contrast=100.0,
    brightness=0.0,
    red_gain=100.0,
    cyan_gain=100.0,
    ghost_reduction=0.0,
    linearize_srgb=False,
):
    return {
        "convergence_px": clamp_float(convergence_px, 0.0, -400.0, 400.0),
        "trim_edges": bool(trim_edges),
        "vertical_offset_px": clamp_float(vertical_offset_px, 0.0, -100.0, 100.0),
        "swap_eyes": bool(swap_eyes),
        "left_gain": clamp_float(left_gain, 100.0, 0.0, 200.0) / 100.0,
        "right_gain": clamp_float(right_gain, 100.0, 0.0, 200.0) / 100.0,
        "saturation": clamp_float(saturation, 100.0, 0.0, 200.0) / 100.0,
        "contrast": clamp_float(contrast, 100.0, 0.0, 200.0) / 100.0,
        "brightness": clamp_float(brightness, 0.0, -100.0, 100.0) / 200.0,
        "red_gain": clamp_float(red_gain, 100.0, 0.0, 200.0) / 100.0,
        "cyan_gain": clamp_float(cyan_gain, 100.0, 0.0, 200.0) / 100.0,
        "ghost_reduction": clamp_float(ghost_reduction, 0.0, 0.0, 100.0) / 100.0,
        "linearize_srgb": bool(linearize_srgb),
    }


def settings_to_advanced(settings):
    return advanced_values(
        convergence_px=settings.get("convergence", 0.0),
        trim_edges=bool_setting(settings, "trim_edges", True),
        vertical_offset_px=settings.get("vertical_offset", 0.0),
        swap_eyes=bool_setting(settings, "swap_eyes", False),
        left_gain=settings.get("left_gain", 100.0),
        right_gain=settings.get("right_gain", 100.0),
        saturation=settings.get("saturation", 100.0),
        contrast=settings.get("contrast", 100.0),
        brightness=settings.get("brightness", 0.0),
        red_gain=settings.get("red_gain", 100.0),
        cyan_gain=settings.get("cyan_gain", 100.0),
        ghost_reduction=settings.get("ghost_reduction", 0.0),
        linearize_srgb=bool_setting(settings, "linearize_srgb", False),
    )


def dubois_anaglyph(
    left_bgr,
    right_bgr,
    convergence_px=0.0,
    trim_edges=True,
    vertical_offset_px=0.0,
    swap_eyes=False,
    left_gain=100.0,
    right_gain=100.0,
    saturation=100.0,
    contrast=100.0,
    brightness=0.0,
    red_gain=100.0,
    cyan_gain=100.0,
    ghost_reduction=0.0,
    linearize_srgb=False,
):
    adv = advanced_values(
        convergence_px,
        trim_edges,
        vertical_offset_px,
        swap_eyes,
        left_gain,
        right_gain,
        saturation,
        contrast,
        brightness,
        red_gain,
        cyan_gain,
        ghost_reduction,
        linearize_srgb,
    )
    left_bgr, right_bgr = resize_to_match(left_bgr, right_bgr)
    if adv["swap_eyes"]:
        left_bgr, right_bgr = right_bgr, left_bgr

    half_shift = adv["convergence_px"] / 2.0
    half_vertical = adv["vertical_offset_px"] / 2.0
    left_bgr = shift_image(left_bgr, half_shift, half_vertical)
    right_bgr = shift_image(right_bgr, -half_shift, -half_vertical)

    left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    left = preprocess_rgb_np(left, adv["left_gain"], adv["saturation"], adv["contrast"], adv["brightness"])
    right = preprocess_rgb_np(right, adv["right_gain"], adv["saturation"], adv["contrast"], adv["brightness"])
    if adv["linearize_srgb"]:
        left = srgb_to_linear_np(left)
        right = srgb_to_linear_np(right)

    out = np.empty_like(left)
    out[..., 0] = (
        DUBOIS_MATRIX[0, 0] * left[..., 0]
        + DUBOIS_MATRIX[0, 1] * left[..., 1]
        + DUBOIS_MATRIX[0, 2] * left[..., 2]
        + DUBOIS_MATRIX[0, 3] * right[..., 0]
        + DUBOIS_MATRIX[0, 4] * right[..., 1]
        + DUBOIS_MATRIX[0, 5] * right[..., 2]
    )
    out[..., 1] = (
        DUBOIS_MATRIX[1, 0] * left[..., 0]
        + DUBOIS_MATRIX[1, 1] * left[..., 1]
        + DUBOIS_MATRIX[1, 2] * left[..., 2]
        + DUBOIS_MATRIX[1, 3] * right[..., 0]
        + DUBOIS_MATRIX[1, 4] * right[..., 1]
        + DUBOIS_MATRIX[1, 5] * right[..., 2]
    )
    out[..., 2] = (
        DUBOIS_MATRIX[2, 0] * left[..., 0]
        + DUBOIS_MATRIX[2, 1] * left[..., 1]
        + DUBOIS_MATRIX[2, 2] * left[..., 2]
        + DUBOIS_MATRIX[2, 3] * right[..., 0]
        + DUBOIS_MATRIX[2, 4] * right[..., 1]
        + DUBOIS_MATRIX[2, 5] * right[..., 2]
    )

    np.clip(out, 0.0, 1.0, out=out)
    if adv["ghost_reduction"] > 0.001:
        amount = adv["ghost_reduction"]
        gray = (
            out[..., 0:1] * 0.2126
            + out[..., 1:2] * 0.7152
            + out[..., 2:3] * 0.0722
        )
        out = gray + (out - gray) * (1.0 - 0.45 * amount)
        out = 0.5 + (out - 0.5) * (1.0 - 0.25 * amount)
        np.clip(out, 0.0, 1.0, out=out)
    out[..., 0] *= adv["red_gain"]
    out[..., 1] *= adv["cyan_gain"]
    out[..., 2] *= adv["cyan_gain"]
    np.clip(out, 0.0, 1.0, out=out)
    if adv["linearize_srgb"]:
        out = linear_to_srgb_np(out)
        np.clip(out, 0.0, 1.0, out=out)
    out_bgr = cv2.cvtColor((out * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if adv["trim_edges"]:
        out_bgr = trim_and_restore_width(out_bgr, adv["convergence_px"])
    return out_bgr


def preprocess_rgb_torch(rgb, gain, saturation, contrast, brightness, torch):
    rgb = rgb * gain
    if abs(saturation - 1.0) > 0.001:
        gray = (
            rgb[..., 0:1] * 0.2126
            + rgb[..., 1:2] * 0.7152
            + rgb[..., 2:3] * 0.0722
        )
        rgb = gray + (rgb - gray) * saturation
    if abs(contrast - 1.0) > 0.001:
        rgb = (rgb - 0.5) * contrast + 0.5
    if abs(brightness) > 0.001:
        rgb = rgb + brightness
    return torch.clamp(rgb, 0.0, 1.0)


def srgb_to_linear_torch(values, torch):
    return torch.where(values <= 0.04045, values / 12.92, torch.pow((values + 0.055) / 1.055, 2.4))


def linear_to_srgb_torch(values, torch):
    return torch.where(values <= 0.0031308, values * 12.92, 1.055 * torch.pow(values, 1.0 / 2.4) - 0.055)


def dubois_anaglyph_torch(left_bgr, right_bgr, adv):
    torch = get_torch_module()
    if torch is None or not torch.cuda.is_available():
        raise ValueError("Torch CUDA 当前不可用。")

    left_bgr, right_bgr = resize_to_match(left_bgr, right_bgr)
    if adv["swap_eyes"]:
        left_bgr, right_bgr = right_bgr, left_bgr

    half_shift = adv["convergence_px"] / 2.0
    half_vertical = adv["vertical_offset_px"] / 2.0
    left_bgr = shift_image(left_bgr, half_shift, half_vertical)
    right_bgr = shift_image(right_bgr, -half_shift, -half_vertical)

    with torch.inference_mode():
        device = torch.device("cuda")
        left = torch.as_tensor(left_bgr, device=device, dtype=torch.float32) / 255.0
        right = torch.as_tensor(right_bgr, device=device, dtype=torch.float32) / 255.0
        left = left[..., [2, 1, 0]]
        right = right[..., [2, 1, 0]]
        left = preprocess_rgb_torch(
            left,
            adv["left_gain"],
            adv["saturation"],
            adv["contrast"],
            adv["brightness"],
            torch,
        )
        right = preprocess_rgb_torch(
            right,
            adv["right_gain"],
            adv["saturation"],
            adv["contrast"],
            adv["brightness"],
            torch,
        )
        if adv["linearize_srgb"]:
            left = srgb_to_linear_torch(left, torch)
            right = srgb_to_linear_torch(right, torch)
        matrix = torch.as_tensor(DUBOIS_MATRIX, device=device)
        out = torch.empty_like(left)
        out[..., 0] = (
            matrix[0, 0] * left[..., 0]
            + matrix[0, 1] * left[..., 1]
            + matrix[0, 2] * left[..., 2]
            + matrix[0, 3] * right[..., 0]
            + matrix[0, 4] * right[..., 1]
            + matrix[0, 5] * right[..., 2]
        )
        out[..., 1] = (
            matrix[1, 0] * left[..., 0]
            + matrix[1, 1] * left[..., 1]
            + matrix[1, 2] * left[..., 2]
            + matrix[1, 3] * right[..., 0]
            + matrix[1, 4] * right[..., 1]
            + matrix[1, 5] * right[..., 2]
        )
        out[..., 2] = (
            matrix[2, 0] * left[..., 0]
            + matrix[2, 1] * left[..., 1]
            + matrix[2, 2] * left[..., 2]
            + matrix[2, 3] * right[..., 0]
            + matrix[2, 4] * right[..., 1]
            + matrix[2, 5] * right[..., 2]
        )
        out = torch.clamp(out, 0.0, 1.0)
        if adv["ghost_reduction"] > 0.001:
            amount = adv["ghost_reduction"]
            gray = (
                out[..., 0:1] * 0.2126
                + out[..., 1:2] * 0.7152
                + out[..., 2:3] * 0.0722
            )
            out = gray + (out - gray) * (1.0 - 0.45 * amount)
            out = 0.5 + (out - 0.5) * (1.0 - 0.25 * amount)
            out = torch.clamp(out, 0.0, 1.0)
        out[..., 0] *= adv["red_gain"]
        out[..., 1] *= adv["cyan_gain"]
        out[..., 2] *= adv["cyan_gain"]
        out = torch.clamp(out, 0.0, 1.0)
        if adv["linearize_srgb"]:
            out = torch.clamp(linear_to_srgb_torch(out, torch), 0.0, 1.0)
        bgr = out[..., [2, 1, 0]]
        out_bgr = (bgr * 255.0 + 0.5).to(dtype=torch.uint8).cpu().numpy()
    if adv["trim_edges"]:
        out_bgr = trim_and_restore_width(out_bgr, adv["convergence_px"])
    return out_bgr


def resolve_backend(settings, job=None):
    requested = settings.get("processing_backend", "auto")
    hardware = detect_hardware()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if hardware.get("torch_cuda"):
            return "torch_cuda"
        if job:
            job.log("已请求 CUDA，但当前未检测到可用 Torch CUDA，自动改用 CPU。")
        return "cpu"
    return "torch_cuda" if hardware.get("torch_cuda") else "cpu"


def process_anaglyph(left_bgr, right_bgr, settings, job=None, backend=None):
    adv = settings_to_advanced(settings)
    backend = backend or resolve_backend(settings, job)
    if backend == "torch_cuda":
        try:
            return dubois_anaglyph_torch(left_bgr, right_bgr, adv)
        except Exception as exc:
            if job:
                job.log(f"CUDA 处理失败，已回退 CPU：{exc}")
    return dubois_anaglyph(
        left_bgr,
        right_bgr,
        convergence_px=adv["convergence_px"],
        trim_edges=adv["trim_edges"],
        vertical_offset_px=adv["vertical_offset_px"],
        swap_eyes=adv["swap_eyes"],
        left_gain=adv["left_gain"] * 100.0,
        right_gain=adv["right_gain"] * 100.0,
        saturation=adv["saturation"] * 100.0,
        contrast=adv["contrast"] * 100.0,
        brightness=adv["brightness"] * 200.0,
        red_gain=adv["red_gain"] * 100.0,
        cyan_gain=adv["cyan_gain"] * 100.0,
        ghost_reduction=adv["ghost_reduction"] * 100.0,
        linearize_srgb=adv["linearize_srgb"],
    )


def split_sbs_frame(frame, mode):
    h, w = frame.shape[:2]
    if w < 2:
        raise ValueError("SBS 画面宽度太小，无法拆分左右眼。")
    half = w // 2
    left = frame[:, :half]
    right = frame[:, half : half * 2]
    if mode in (VIDEO_HALF_SBS, PHOTO_HALF_SBS):
        left = cv2.resize(left, (w, h), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, (w, h), interpolation=cv2.INTER_LINEAR)
    return left, right


def capture_video_frame(path, percent):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 1:
            index = int(round((max(0.0, min(100.0, percent)) / 100.0) * (frame_count - 1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"无法读取视频帧：{path}")
        return frame
    finally:
        cap.release()


def inspect_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        return {
            "path": str(path),
            "fps": fps,
            "frames": frame_count,
            "duration": duration,
            "duration_label": format_duration(duration),
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


def capture_video_frame_at_time(path, seconds):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{path}")
    try:
        seconds = max(0.0, float(seconds or 0.0))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps > 0 and frame_count > 1:
            index = min(frame_count - 1, max(0, int(round(seconds * fps))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"无法读取视频帧：{path}")
        return frame
    finally:
        cap.release()


def inspect_video_settings(settings):
    mode = settings.get("video_mode")
    infos = []
    if mode == VIDEO_SEPARATE:
        if settings.get("video_left"):
            infos.append(inspect_video(settings["video_left"]))
        if settings.get("video_right"):
            infos.append(inspect_video(settings["video_right"]))
    elif settings.get("video_sbs"):
        infos.append(inspect_video(settings["video_sbs"]))

    if not infos:
        return {"ok": True, "duration": 0.0, "duration_label": "00:00", "videos": []}

    durations = [info["duration"] for info in infos if info["duration"] > 0]
    duration = min(durations) if durations else 0.0
    return {
        "ok": True,
        "duration": duration,
        "duration_label": format_duration(duration),
        "videos": infos,
    }


def sequence_key(path):
    stem = Path(path).stem
    match = re.search(r"(\d+)(?!.*\d)", stem)
    if match:
        digits = match.group(1)
        return ("number", int(digits), digits)
    return ("name", stem.lower(), stem)


def discover_sequence_pairs(folder, left_pattern, right_pattern):
    folder = Path(folder)
    left_files = sorted(glob.glob(str(folder / left_pattern)))
    right_files = sorted(glob.glob(str(folder / right_pattern)))
    if not left_files or not right_files:
        return []

    left_map = {sequence_key(path)[:2]: path for path in left_files}
    right_map = {sequence_key(path)[:2]: path for path in right_files}
    shared = sorted(left_map.keys() & right_map.keys(), key=lambda value: (str(value[0]), value[1]))
    if shared:
        pairs = []
        for key in shared:
            pretty = sequence_key(left_map[key])[2]
            pairs.append((pretty, left_map[key], right_map[key]))
        return pairs

    count = min(len(left_files), len(right_files))
    pairs = []
    for index in range(count):
        pairs.append((f"{index + 1:04d}", left_files[index], right_files[index]))
    return pairs


def safe_stem(text):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("._") or "frame"


def fit_preview(image_bgr, max_side=1600):
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return image_bgr
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)


class JobState:
    def __init__(self):
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.thread = None
        self.running = False
        self.progress = 0.0
        self.text = "准备就绪"
        self.logs = []

    def log(self, message):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        with self.lock:
            self.logs.append(line)
            self.logs = self.logs[-300:]
            self.text = message

    def set_progress(self, value, text=None):
        with self.lock:
            self.progress = max(0.0, min(100.0, float(value)))
            if text is not None:
                self.text = text

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "progress": self.progress,
                "text": self.text,
                "logs": list(self.logs),
            }

    def start(self, target, settings):
        with self.lock:
            if self.running:
                raise ValueError("当前已有转换任务在运行。")
            self.running = True
            self.progress = 0.0
            self.text = "开始处理..."
            self.logs = []
            self.cancel_event.clear()
        self.thread = threading.Thread(target=self._run, args=(target, settings), daemon=True)
        self.thread.start()

    def _run(self, target, settings):
        started = time.time()
        try:
            target(settings, self)
            elapsed = time.time() - started
            if self.cancel_event.is_set():
                self.log(f"任务已停止，用时 {elapsed:.1f} 秒。")
            else:
                self.set_progress(100.0, f"转换完成，用时 {elapsed:.1f} 秒。")
                self.log(f"转换完成，用时 {elapsed:.1f} 秒。")
        except Exception as exc:
            self.log(f"错误：{exc}")
            self.log(traceback.format_exc())
            self.set_progress(0.0, "任务失败")
        finally:
            with self.lock:
                self.running = False


JOB = JobState()


def report_progress(job, done, total, every=1):
    if done % every != 0 and done != total:
        return
    if total:
        percent = min(100.0, done * 100.0 / total)
        text = f"处理中：{done}/{total}（{percent:.1f}%）"
    else:
        percent = 0.0
        text = f"处理中：{done} 帧"
    job.set_progress(percent, text)


def ensure_video_writer(writer, output, fps, frame_shape, job):
    if writer is not None:
        return writer
    h, w = frame_shape
    ext = Path(output).suffix.lower()
    codecs = ("XVID", "MJPG") if ext == ".avi" else ("mp4v", "avc1", "MJPG")
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(str(output), fourcc, float(fps), (w, h))
        if candidate.isOpened():
            job.log(f"视频编码器：{codec}，输出尺寸：{w}x{h}。")
            return candidate
        candidate.release()
    raise ValueError("无法创建视频输出文件。请尝试改用 .avi，或安装支持 MP4 的 OpenCV/FFmpeg 环境。")


def selected_audio_source(settings):
    if settings.get("video_mode") == VIDEO_SEPARATE:
        source = settings.get("audio_source", "left")
        if source == "right":
            return settings.get("video_right")
        if source == "none":
            return None
        return settings.get("video_left")
    return settings.get("video_sbs")


def remux_audio(video_only_path, audio_source_path, output_path, job):
    hardware = detect_hardware()
    ffmpeg = hardware.get("ffmpeg")
    if not ffmpeg:
        job.log("未找到 FFmpeg，无法自动合并音频；已保留无声音画面视频。")
        return False
    if not audio_source_path:
        job.log("未选择音频来源，已输出无声视频。")
        return False

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_only_path),
        "-i",
        str(audio_source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(output_path),
    ]
    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        startupinfo=startupinfo,
    )
    if completed.returncode != 0:
        job.log("FFmpeg 音频合并失败，已保留无声视频。")
        job.log(completed.stderr[-1200:])
        return False
    job.log("已使用 FFmpeg 复用原始音频轨道。")
    return True


def prepare_video_output(settings, output):
    include_audio = bool_setting(settings, "include_audio", True)
    audio_source = selected_audio_source(settings)
    hardware = detect_hardware()
    if include_audio and audio_source and hardware.get("ffmpeg"):
        temp_output = output.with_name(f"{output.stem}.video_only{output.suffix}")
        if temp_output == output:
            temp_output = output.with_name(f"{output.stem}.tmp{output.suffix}")
        return temp_output, True, audio_source
    return output, False, audio_source


def convert_video_worker(settings, job):
    mode = settings["video_mode"]
    output = normalize_output_path(settings["video_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    video_output, needs_audio_mux, audio_source = prepare_video_output(settings, output)
    backend = resolve_backend(settings, job)
    job.log(f"处理后端：{'Torch CUDA' if backend == 'torch_cuda' else 'CPU / NumPy'}。")
    if bool_setting(settings, "include_audio", True) and not detect_hardware().get("ffmpeg"):
        job.log("未检测到 FFmpeg：视频可正常输出，但无法自动带上声音。")

    if mode == VIDEO_SEPARATE:
        left_cap = cv2.VideoCapture(settings["video_left"])
        right_cap = cv2.VideoCapture(settings["video_right"])
        if not left_cap.isOpened() or not right_cap.isOpened():
            left_cap.release()
            right_cap.release()
            raise ValueError("无法打开左右眼视频。")
        try:
            fps = left_cap.get(cv2.CAP_PROP_FPS) or 24.0
            left_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            right_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            total = min(left_count, right_count) if left_count and right_count else 0
            job.log(f"开始视频转换：左右眼独立视频，FPS {fps:.3g}，预计 {total or '未知'} 帧。")
            writer = None
            index = 0
            while not job.cancel_event.is_set():
                ok_l, left = left_cap.read()
                ok_r, right = right_cap.read()
                if not ok_l or not ok_r:
                    break
                output_frame = process_anaglyph(left, right, settings, job, backend)
                writer = ensure_video_writer(writer, video_output, fps, output_frame.shape[:2], job)
                writer.write(output_frame)
                index += 1
                report_progress(job, index, total, every=5)
            if writer:
                writer.release()
            if needs_audio_mux and not job.cancel_event.is_set():
                ok = remux_audio(video_output, audio_source, output, job)
                if ok:
                    try:
                        video_output.unlink(missing_ok=True)
                    except TypeError:
                        if video_output.exists():
                            video_output.unlink()
                else:
                    job.log(f"无声音画面视频保存在：{video_output}")
            job.log(f"已写入 {index} 帧。")
        finally:
            left_cap.release()
            right_cap.release()
        return

    cap = cv2.VideoCapture(settings["video_sbs"])
    if not cap.isOpened():
        raise ValueError("无法打开 SBS 视频。")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        label = "Full SBS 视频" if mode == VIDEO_FULL_SBS else "Half SBS 视频"
        job.log(f"开始视频转换：{label}，FPS {fps:.3g}，预计 {total or '未知'} 帧。")
        writer = None
        index = 0
        while not job.cancel_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            left, right = split_sbs_frame(frame, mode)
            output_frame = process_anaglyph(left, right, settings, job, backend)
            writer = ensure_video_writer(writer, video_output, fps, output_frame.shape[:2], job)
            writer.write(output_frame)
            index += 1
            report_progress(job, index, total, every=5)
        if writer:
            writer.release()
        if needs_audio_mux and not job.cancel_event.is_set():
            ok = remux_audio(video_output, audio_source, output, job)
            if ok:
                try:
                    video_output.unlink(missing_ok=True)
                except TypeError:
                    if video_output.exists():
                        video_output.unlink()
            else:
                job.log(f"无声音画面视频保存在：{video_output}")
        job.log(f"已写入 {index} 帧。")
    finally:
        cap.release()


def convert_sequence_worker(settings, job):
    pairs = discover_sequence_pairs(
        settings["sequence_folder"],
        settings.get("sequence_left_pattern") or "left_*.*",
        settings.get("sequence_right_pattern") or "right_*.*",
    )
    if not pairs:
        raise ValueError("没有找到可配对的左右眼图片。")
    output_dir = normalize_output_path(settings["sequence_output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = settings.get("sequence_ext") or ".png"
    if not ext.startswith("."):
        ext = "." + ext
    total = len(pairs)
    backend = resolve_backend(settings, job)
    job.log(f"处理后端：{'Torch CUDA' if backend == 'torch_cuda' else 'CPU / NumPy'}。")
    job.log(f"开始转换图片序列：{total} 组。")
    for index, (key, left_path, right_path) in enumerate(pairs, start=1):
        if job.cancel_event.is_set():
            break
        left = read_image(left_path)
        right = read_image(right_path)
        result = process_anaglyph(left, right, settings, job, backend)
        output_path = output_dir / f"anaglyph_{safe_stem(key)}{ext}"
        write_image(output_path, result)
        report_progress(job, index, total, every=1)
    job.log(f"图片序列输出文件夹：{output_dir}")


def convert_photo_worker(settings, job):
    mode = settings["photo_mode"]
    if mode == PHOTO_SEPARATE:
        left = read_image(settings["photo_left"])
        right = read_image(settings["photo_right"])
    else:
        frame = read_image(settings["photo_sbs"])
        left, right = split_sbs_frame(frame, mode)
    backend = resolve_backend(settings, job)
    job.log(f"处理后端：{'Torch CUDA' if backend == 'torch_cuda' else 'CPU / NumPy'}。")
    result = process_anaglyph(left, right, settings, job, backend)
    output = normalize_output_path(settings["photo_output"])
    write_image(output, result)
    job.set_progress(100.0, "单张照片转换完成")
    job.log(f"照片输出：{output}")


def load_preview_pair(task, settings):
    percent = float(settings.get("preview_percent", 0.0))
    seconds = float(settings.get("preview_seconds", 0.0) or 0.0)
    if task == "video":
        mode = settings.get("video_mode")
        if mode == VIDEO_SEPARATE:
            left_path = settings.get("video_left")
            right_path = settings.get("video_right")
            if not left_path or not right_path:
                raise ValueError("请选择左眼视频和右眼视频。")
            return capture_video_frame_at_time(left_path, seconds), capture_video_frame_at_time(right_path, seconds)
        sbs_path = settings.get("video_sbs")
        if not sbs_path:
            raise ValueError("请选择 SBS 视频。")
        frame = capture_video_frame_at_time(sbs_path, seconds)
        return split_sbs_frame(frame, mode)

    if task == "sequence":
        folder = settings.get("sequence_folder")
        if not folder:
            raise ValueError("请选择图片序列文件夹。")
        pairs = discover_sequence_pairs(
            folder,
            settings.get("sequence_left_pattern") or "left_*.*",
            settings.get("sequence_right_pattern") or "right_*.*",
        )
        if not pairs:
            raise ValueError("没有找到可配对的左右眼图片。")
        index = int(round((percent / 100.0) * (len(pairs) - 1)))
        _key, left_path, right_path = pairs[max(0, min(index, len(pairs) - 1))]
        return read_image(left_path), read_image(right_path)

    if task == "photo":
        mode = settings.get("photo_mode")
        if mode == PHOTO_SEPARATE:
            left_path = settings.get("photo_left")
            right_path = settings.get("photo_right")
            if not left_path or not right_path:
                raise ValueError("请选择左眼照片和右眼照片。")
            return read_image(left_path), read_image(right_path)
        sbs_path = settings.get("photo_sbs")
        if not sbs_path:
            raise ValueError("请选择 SBS 照片。")
        frame = read_image(sbs_path)
        return split_sbs_frame(frame, mode)

    raise ValueError("未知预览类型。")


def validate_start_payload(task, settings):
    if task == "video":
        mode = settings.get("video_mode")
        if mode == VIDEO_SEPARATE:
            if not settings.get("video_left") or not settings.get("video_right"):
                raise ValueError("请先选择左眼视频和右眼视频。")
        elif not settings.get("video_sbs"):
            raise ValueError("请先选择 SBS 视频。")
        if not settings.get("video_output"):
            raise ValueError("请选择输出视频路径。")
        return convert_video_worker

    if task == "sequence":
        if not settings.get("sequence_folder") or not settings.get("sequence_output"):
            raise ValueError("请选择输入文件夹和输出文件夹。")
        return convert_sequence_worker

    if task == "photo":
        mode = settings.get("photo_mode")
        if mode == PHOTO_SEPARATE:
            if not settings.get("photo_left") or not settings.get("photo_right"):
                raise ValueError("请先选择左眼照片和右眼照片。")
        elif not settings.get("photo_sbs"):
            raise ValueError("请先选择 SBS 照片。")
        if not settings.get("photo_output"):
            raise ValueError("请选择输出照片路径。")
        return convert_photo_worker

    raise ValueError("未知任务类型。")


def powershell_dialog(script):
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-STA",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or "文件选择器打开失败。")
    return completed.stdout.strip()


def pick_path(kind):
    video_filter = "视频文件|*.mp4;*.mov;*.avi;*.mkv;*.m4v|所有文件|*.*"
    image_filter = "图片文件|*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp|所有文件|*.*"
    if kind == "folder":
        script = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '选择文件夹'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  Write-Output $dialog.SelectedPath
}
"""
        return powershell_dialog(script)

    if kind in ("open_video", "open_image"):
        title = "选择视频" if kind == "open_video" else "选择图片"
        filter_text = video_filter if kind == "open_video" else image_filter
        script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = '{title}'
$dialog.Filter = '{filter_text}'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  Write-Output $dialog.FileName
}}
"""
        return powershell_dialog(script)

    if kind in ("save_video", "save_image"):
        title = "保存视频" if kind == "save_video" else "保存图片"
        filter_text = (
            "MP4 视频|*.mp4|AVI 视频|*.avi|MOV 视频|*.mov|所有文件|*.*"
            if kind == "save_video"
            else "PNG 图片|*.png|JPEG 图片|*.jpg|TIFF 图片|*.tif|所有文件|*.*"
        )
        default_ext = "mp4" if kind == "save_video" else "png"
        script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.SaveFileDialog
$dialog.Title = '{title}'
$dialog.Filter = '{filter_text}'
$dialog.DefaultExt = '{default_ext}'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  Write-Output $dialog.FileName
}}
"""
        return powershell_dialog(script)

    raise ValueError("未知选择器类型。")


def open_output_location(path):
    target = normalize_output_path(path) if path else OUTPUT_DIR
    if target.suffix:
        target.parent.mkdir(parents=True, exist_ok=True)
        target = target.parent
    else:
        target.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        subprocess.Popen(["explorer", str(target)])
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, str(target)])
    return str(target)


def crossref_search(query, rows=3):
    url = f"https://api.crossref.org/works?query={quote_plus(query)}&rows={int(rows)}"
    request = urllib.request.Request(url, headers={"User-Agent": "DuboisAnaglyphTool/1.0"})
    with urllib.request.urlopen(request, timeout=8) as response:
        data = json.loads(response.read().decode("utf-8"))
    items = data.get("message", {}).get("items", [])
    results = []
    for item in items:
        title = (item.get("title") or [""])[0]
        if not title:
            continue
        year = ""
        for key in ("published-print", "published-online", "created"):
            parts = item.get(key, {}).get("date-parts")
            if parts and parts[0]:
                year = str(parts[0][0])
                break
        doi = item.get("DOI")
        link = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
        authors = []
        for author in item.get("author", [])[:3]:
            name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
            if name:
                authors.append(name)
        results.append(
            {
                "title": title,
                "authors": ", ".join(authors) or "Crossref",
                "year": year,
                "url": link,
                "note": "自动搜索结果，可用于继续查阅立体显示、串扰和舒适度调节。",
            }
        )
    return results


def search_papers():
    queries = [
        "Dubois anaglyph stereo images projection method",
        "stereoscopic display crosstalk visual comfort disparity",
        "anaglyph stereo retinal rivalry ghosting",
    ]
    found = []
    seen = set()
    for query in queries:
        try:
            for item in crossref_search(query, rows=3):
                key = (item.get("title") or "").lower()
                if key and key not in seen:
                    seen.add(key)
                    found.append(item)
        except Exception:
            continue
    if not found:
        found = list(PAPER_REFERENCES)
    return {
        "ok": True,
        "results": found[:8],
        "curated": PAPER_REFERENCES,
        "suggestions": [
            "先用汇聚和垂直校正把主要主体对齐，再微调颜色。",
            "鬼影明显时，提高鬼影抑制或略降对比度/饱和度。",
            "红青眼镜偏色明显时，用红通道增益和青通道增益做眼镜适配。",
            "长片段优先开启自动硬件加速；没有 CUDA 时 CPU 会自动接管。",
        ],
    }


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dubois 红青立体转换器</title>
  <style>
    :root {
      --bg: #edf1f4;
      --panel: #ffffff;
      --panel-2: #f8fafb;
      --text: #17212b;
      --muted: #667887;
      --line: #d9e2e8;
      --line-strong: #bfccd5;
      --red: #d74d59;
      --cyan: #009bad;
      --cyan-dark: #087887;
      --ink: #0e151b;
      --soft: #f3f6f8;
      --focus: rgba(0, 155, 173, 0.18);
      --shadow: 0 18px 46px rgba(21, 31, 40, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(180deg, #f8fafb 0, var(--bg) 280px),
        var(--bg);
      color: var(--text);
      font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
      font-size: 14px;
    }
    .app {
      width: min(1500px, calc(100vw - 32px));
      margin: 18px auto;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
      padding: 16px 18px;
      background: rgba(255, 255, 255, 0.84);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 10px 30px rgba(21, 31, 40, 0.07);
      backdrop-filter: blur(12px);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }
    .brand-mark {
      width: 42px;
      height: 42px;
      border-radius: 8px;
      background:
        linear-gradient(90deg, rgba(215, 77, 89, 0.92) 0 46%, rgba(0, 155, 173, 0.92) 54% 100%),
        #101820;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.36), 0 10px 24px rgba(0, 0, 0, 0.12);
      flex: 0 0 auto;
    }
    .brand-copy {
      min-width: 0;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
      font-weight: 760;
      letter-spacing: 0;
    }
    .subtitle {
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 5px 10px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--muted);
      border-radius: 6px;
      white-space: nowrap;
      box-shadow: 0 4px 12px rgba(21, 31, 40, 0.04);
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(520px, 1.1fr) minmax(420px, 0.9fr);
      gap: 14px;
      align-items: stretch;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .controls {
      min-height: 590px;
      display: flex;
      flex-direction: column;
    }
    .tabs {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 8px 8px 0 0;
      overflow: hidden;
    }
    .tab-button {
      border: 0;
      border-right: 1px solid var(--line);
      background: transparent;
      color: var(--muted);
      height: 48px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    .tab-button:last-child { border-right: 0; }
    .tab-button.active {
      background: var(--panel);
      color: var(--text);
      box-shadow: inset 0 -3px 0 var(--cyan), 0 8px 18px rgba(0, 155, 173, 0.06);
    }
    .tab-panel {
      display: none;
      padding: 18px;
    }
    .tab-panel.active { display: block; }
    .section-title {
      font-size: 14px;
      font-weight: 760;
      margin: 0 0 14px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--line);
      color: #263541;
    }
    .grid {
      display: grid;
      grid-template-columns: 128px minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }
    label {
      color: var(--muted);
      font-weight: 650;
    }
    input[type="text"], select, input[type="number"] {
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--text);
      background: #fff;
      font: inherit;
      outline: none;
    }
    input[type="text"]:focus, select:focus, input[type="number"]:focus {
      border-color: var(--cyan);
      box-shadow: 0 0 0 3px var(--focus);
    }
    select { cursor: pointer; }
    button {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 12px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: border-color 120ms ease, background 120ms ease, transform 120ms ease, box-shadow 120ms ease;
    }
    button:hover {
      border-color: #aab9c4;
      background: #fbfcfd;
      box-shadow: 0 6px 16px rgba(21, 31, 40, 0.08);
    }
    button:active { transform: translateY(1px); }
    button.primary {
      border-color: var(--cyan);
      background: var(--cyan);
      color: #fff;
      box-shadow: 0 8px 18px rgba(0, 155, 173, 0.20);
    }
    button.primary:hover { background: var(--cyan-dark); }
    button.danger {
      border-color: #efb5ba;
      color: #a52f38;
    }
    .button-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 16px;
    }
    .mode-group { display: none; }
    .mode-group.active { display: block; }
    .preview-panel {
      display: grid;
      grid-template-rows: auto minmax(340px, 1fr) auto;
      min-height: 590px;
      overflow: hidden;
    }
    .preview-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
    }
    .preview-title {
      font-weight: 760;
      font-size: 15px;
    }
    .convergence-readout {
      color: var(--red);
      font-weight: 800;
      white-space: nowrap;
    }
    .stage {
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      background:
        linear-gradient(135deg, rgba(215, 77, 89, 0.12), rgba(0, 155, 173, 0.14)),
        var(--ink);
      min-height: 340px;
      overflow: hidden;
    }
    .stage img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: none;
    }
    .placeholder {
      color: #cbd5dc;
      text-align: center;
      padding: 24px;
      line-height: 1.7;
    }
    .adjust {
      padding: 14px 16px 16px;
      border-top: 1px solid var(--line);
      background: var(--panel-2);
    }
    .slider-row {
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr) 80px;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--cyan);
    }
    .check-row {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-weight: 650;
    }
    .bottom {
      margin-top: 12px;
      padding: 14px;
    }
    .progress-line {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
    }
    progress {
      width: 100%;
      height: 14px;
      accent-color: var(--cyan);
    }
    .log {
      height: 150px;
      overflow: auto;
      white-space: pre-wrap;
      background: #101418;
      color: #d9e2e8;
      border-radius: 6px;
      padding: 10px;
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 12px;
      line-height: 1.55;
    }
    .compact-row {
      display: grid;
      grid-template-columns: 128px minmax(0, 1fr) 120px minmax(0, 1fr) 92px;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }
    .advanced {
      border-top: 1px solid var(--line);
      padding: 16px 18px 18px;
      background: var(--panel-2);
    }
    .advanced-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(240px, 1fr));
      gap: 10px 16px;
      align-items: center;
    }
    .advanced-row {
      display: grid;
      grid-template-columns: 108px minmax(0, 1fr) 72px;
      align-items: center;
      gap: 8px;
    }
    .wide-row {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .paper-list {
      grid-column: 1 / -1;
      max-height: 145px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 8px 10px;
      color: var(--muted);
      line-height: 1.55;
    }
    .paper-list a {
      color: #087887;
      font-weight: 700;
      text-decoration: none;
    }
    .paper-list a:hover { text-decoration: underline; }
    @media (max-width: 980px) {
      .workspace { grid-template-columns: 1fr; }
      .compact-row, .grid, .advanced-grid, .advanced-row { grid-template-columns: 1fr; }
      .grid button, .compact-row button { width: 100%; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <div class="brand-mark" aria-hidden="true"></div>
        <div class="brand-copy">
          <h1>Dubois 红青立体转换器</h1>
          <div class="subtitle">立体视频与图片序列转红青 · 实时预览 · GPU 自动加速</div>
        </div>
      </div>
      <div class="badge" id="statusBadge">本地运行 · OpenCV 处理</div>
    </header>

    <main class="workspace">
      <section class="panel controls">
        <nav class="tabs">
          <button class="tab-button active" data-tab="video">视频模式</button>
          <button class="tab-button" data-tab="sequence">图片序列</button>
          <button class="tab-button" data-tab="photo">单张照片</button>
        </nav>

        <div class="tab-panel active" id="tab-video">
          <p class="section-title">输入与输出</p>
          <div class="grid">
            <label for="videoMode">输入类型</label>
            <select id="videoMode">
              <option value="separate">左右眼独立视频</option>
              <option value="full_sbs">Full SBS</option>
              <option value="half_sbs">Half SBS</option>
            </select>
            <span></span>
          </div>
          <div class="mode-group active" id="videoSeparate">
            <div class="grid">
              <label for="videoLeft">左眼视频</label>
              <input id="videoLeft" type="text">
              <button data-pick="open_video" data-target="videoLeft">浏览</button>
            </div>
            <div class="grid">
              <label for="videoRight">右眼视频</label>
              <input id="videoRight" type="text">
              <button data-pick="open_video" data-target="videoRight">浏览</button>
            </div>
          </div>
          <div class="mode-group" id="videoSbs">
            <div class="grid">
              <label for="videoSbsPath">SBS 视频</label>
              <input id="videoSbsPath" type="text">
              <button data-pick="open_video" data-target="videoSbsPath">浏览</button>
            </div>
          </div>
          <div class="grid">
            <label for="videoOutput">输出视频</label>
            <input id="videoOutput" type="text" value="output\\anaglyph_output.mp4">
            <button data-pick="save_video" data-target="videoOutput">浏览</button>
          </div>
          <div class="button-row">
            <button id="previewVideo">刷新预览</button>
            <button class="primary" id="startVideo">开始转换视频</button>
            <button id="openVideoOutput">打开输出文件夹</button>
          </div>
        </div>

        <div class="tab-panel" id="tab-sequence">
          <p class="section-title">输入与输出</p>
          <div class="grid">
            <label for="sequenceFolder">输入文件夹</label>
            <input id="sequenceFolder" type="text">
            <button data-pick="folder" data-target="sequenceFolder">浏览</button>
          </div>
          <div class="grid">
            <label for="sequenceOutput">输出文件夹</label>
            <input id="sequenceOutput" type="text" value="output\\anaglyph_sequence">
            <button data-pick="folder" data-target="sequenceOutput">浏览</button>
          </div>
          <div class="compact-row">
            <label for="sequenceLeftPattern">左眼通配符</label>
            <input id="sequenceLeftPattern" type="text" value="left_*.png">
            <label for="sequenceRightPattern">右眼通配符</label>
            <input id="sequenceRightPattern" type="text" value="right_*.png">
            <select id="sequenceExt">
              <option value=".png">PNG</option>
              <option value=".jpg">JPG</option>
              <option value=".tif">TIF</option>
            </select>
          </div>
          <div class="button-row">
            <button id="scanSequence">扫描序列</button>
            <button id="previewSequence">刷新预览</button>
            <button class="primary" id="startSequence">开始转换序列</button>
            <button id="openSequenceOutput">打开输出文件夹</button>
          </div>
        </div>

        <div class="tab-panel" id="tab-photo">
          <p class="section-title">输入与输出</p>
          <div class="grid">
            <label for="photoMode">输入类型</label>
            <select id="photoMode">
              <option value="photo_separate">左右眼独立照片</option>
              <option value="photo_full_sbs">Full SBS</option>
              <option value="photo_half_sbs">Half SBS</option>
            </select>
            <span></span>
          </div>
          <div class="mode-group active" id="photoSeparate">
            <div class="grid">
              <label for="photoLeft">左眼照片</label>
              <input id="photoLeft" type="text">
              <button data-pick="open_image" data-target="photoLeft">浏览</button>
            </div>
            <div class="grid">
              <label for="photoRight">右眼照片</label>
              <input id="photoRight" type="text">
              <button data-pick="open_image" data-target="photoRight">浏览</button>
            </div>
          </div>
          <div class="mode-group" id="photoSbs">
            <div class="grid">
              <label for="photoSbsPath">SBS 照片</label>
              <input id="photoSbsPath" type="text">
              <button data-pick="open_image" data-target="photoSbsPath">浏览</button>
            </div>
          </div>
          <div class="grid">
            <label for="photoOutput">输出照片</label>
            <input id="photoOutput" type="text" value="output\\anaglyph_photo.png">
            <button data-pick="save_image" data-target="photoOutput">浏览</button>
          </div>
          <div class="button-row">
            <button id="previewPhoto">刷新预览</button>
            <button class="primary" id="startPhoto">转换单张照片</button>
            <button id="openPhotoOutput">打开输出文件夹</button>
          </div>
        </div>

        <div class="advanced">
          <p class="section-title">高级调节、音频与硬件</p>
          <div class="advanced-grid">
            <div class="advanced-row">
              <label for="processingBackend">硬件加速</label>
              <select id="processingBackend">
                <option value="auto">自动判断</option>
                <option value="cuda">强制 CUDA</option>
                <option value="cpu">强制 CPU</option>
              </select>
              <button id="refreshHardware">检测</button>
            </div>
            <div class="advanced-row">
              <label for="audioSource">音频来源</label>
              <select id="audioSource">
                <option value="left">左眼 / SBS 原片</option>
                <option value="right">右眼视频</option>
                <option value="none">不带音频</option>
              </select>
              <label class="check-row"><input id="includeAudio" type="checkbox" checked><span>带音频</span></label>
            </div>
            <div class="wide-row">
              <span class="badge" id="hardwareStatus">等待硬件检测</span>
            </div>
            <div class="advanced-row">
              <label for="verticalOffset">垂直校正</label>
              <input id="verticalOffset" type="range" min="-80" max="80" step="0.1" value="0">
              <input id="verticalOffsetNum" type="number" min="-80" max="80" step="0.1" value="0">
            </div>
            <div class="advanced-row">
              <label for="ghostReduction">鬼影抑制</label>
              <input id="ghostReduction" type="range" min="0" max="100" step="1" value="0">
              <input id="ghostReductionNum" type="number" min="0" max="100" step="1" value="0">
            </div>
            <div class="advanced-row">
              <label for="saturation">饱和度</label>
              <input id="saturation" type="range" min="0" max="180" step="1" value="100">
              <input id="saturationNum" type="number" min="0" max="180" step="1" value="100">
            </div>
            <div class="advanced-row">
              <label for="contrast">对比度</label>
              <input id="contrast" type="range" min="40" max="180" step="1" value="100">
              <input id="contrastNum" type="number" min="40" max="180" step="1" value="100">
            </div>
            <div class="advanced-row">
              <label for="brightness">亮度</label>
              <input id="brightness" type="range" min="-60" max="60" step="1" value="0">
              <input id="brightnessNum" type="number" min="-60" max="60" step="1" value="0">
            </div>
            <div class="advanced-row">
              <label for="leftGain">左眼增益</label>
              <input id="leftGain" type="range" min="40" max="160" step="1" value="100">
              <input id="leftGainNum" type="number" min="40" max="160" step="1" value="100">
            </div>
            <div class="advanced-row">
              <label for="rightGain">右眼增益</label>
              <input id="rightGain" type="range" min="40" max="160" step="1" value="100">
              <input id="rightGainNum" type="number" min="40" max="160" step="1" value="100">
            </div>
            <div class="advanced-row">
              <label for="redGain">红通道</label>
              <input id="redGain" type="range" min="40" max="180" step="1" value="100">
              <input id="redGainNum" type="number" min="40" max="180" step="1" value="100">
            </div>
            <div class="advanced-row">
              <label for="cyanGain">青通道</label>
              <input id="cyanGain" type="range" min="40" max="180" step="1" value="100">
              <input id="cyanGainNum" type="number" min="40" max="180" step="1" value="100">
            </div>
            <div class="wide-row">
              <label class="check-row"><input id="linearizeSrgb" type="checkbox"><span>sRGB 线性化</span></label>
              <label class="check-row"><input id="swapEyes" type="checkbox"><span>交换左右眼</span></label>
              <button id="searchPapers">搜索论文并加载建议</button>
            </div>
            <div class="paper-list" id="paperResults">点击“搜索论文并加载建议”后显示相关论文和调节提示。</div>
          </div>
        </div>
      </section>

      <aside class="panel preview-panel">
        <div class="preview-head">
          <div class="preview-title">实时重叠结果</div>
          <div class="convergence-readout" id="convergenceValue">0.0 px</div>
        </div>
        <div class="stage">
          <img id="previewImage" alt="">
          <div class="placeholder" id="placeholder">选择输入后显示预览</div>
        </div>
        <div class="adjust">
          <div class="slider-row">
            <label for="coarse">粗调</label>
            <input id="coarse" type="range" min="-200" max="200" step="1" value="0">
            <input id="coarseNum" type="number" min="-200" max="200" step="1" value="0">
          </div>
          <div class="slider-row">
            <label for="fine">精调</label>
            <input id="fine" type="range" min="-10" max="10" step="0.1" value="0">
            <input id="fineNum" type="number" min="-10" max="10" step="0.1" value="0">
          </div>
          <div class="slider-row">
            <label for="previewPercent" id="previewPositionLabel">视频时间轴</label>
            <input id="previewPercent" type="range" min="0" max="100" step="0.1" value="0">
            <input id="previewPercentNum" type="number" min="0" max="100" step="0.1" value="0">
          </div>
          <div class="badge" id="timelineReadout">00:00 / 00:00</div>
          <label class="check-row">
            <input id="trimEdges" type="checkbox" checked>
            <span>自动裁边，隐藏汇聚边缘伪影</span>
          </label>
        </div>
      </aside>
    </main>

    <section class="panel bottom">
      <div class="progress-line">
        <progress id="progress" value="0" max="100"></progress>
        <button class="danger" id="cancelTask">停止当前任务</button>
      </div>
      <div class="badge" id="progressText">准备就绪</div>
      <div class="log" id="log"></div>
    </section>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    let activeTab = "video";
    let previewTimer = null;
    let previewUrl = null;
    let videoDuration = 0;

    function formatTime(seconds) {
      const total = Math.max(0, Math.round(Number(seconds || 0)));
      const hours = Math.floor(total / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      const secs = total % 60;
      const two = (value) => String(value).padStart(2, "0");
      return hours > 0 ? `${two(hours)}:${two(minutes)}:${two(secs)}` : `${two(minutes)}:${two(secs)}`;
    }

    function getConvergence() {
      return Number($("coarse").value || 0) + Number($("fine").value || 0);
    }

    function setReadouts() {
      $("coarseNum").value = $("coarse").value;
      $("fineNum").value = Number($("fine").value).toFixed(1);
      if (activeTab === "video") {
        $("previewPercentNum").value = Number($("previewPercent").value || 0).toFixed(1);
        $("timelineReadout").textContent = `${formatTime($("previewPercent").value)} / ${formatTime(videoDuration)}`;
        $("previewPositionLabel").textContent = "视频时间轴";
      } else if (activeTab === "sequence") {
        $("previewPercentNum").value = Math.round(Number($("previewPercent").value || 0));
        $("timelineReadout").textContent = `序列位置 ${Math.round(Number($("previewPercent").value || 0))}%`;
        $("previewPositionLabel").textContent = "序列位置";
      } else {
        $("previewPercentNum").value = "0";
        $("timelineReadout").textContent = "单张照片预览";
        $("previewPositionLabel").textContent = "预览";
      }
      $("convergenceValue").textContent = `${getConvergence().toFixed(1)} px`;
    }

    function syncPair(rangeId, numberId, precision) {
      const range = $(rangeId);
      const number = $(numberId);
      range.addEventListener("input", () => {
        number.value = precision === 1 ? Number(range.value).toFixed(1) : range.value;
        setReadouts();
        schedulePreview();
      });
      number.addEventListener("input", () => {
        const min = Number(range.min);
        const max = Number(range.max);
        const value = Math.max(min, Math.min(max, Number(number.value || 0)));
        range.value = value;
        setReadouts();
        schedulePreview();
      });
    }

    function collectSettings() {
      return {
        video_mode: $("videoMode").value,
        video_left: $("videoLeft").value.trim(),
        video_right: $("videoRight").value.trim(),
        video_sbs: $("videoSbsPath").value.trim(),
        video_output: $("videoOutput").value.trim(),
        sequence_folder: $("sequenceFolder").value.trim(),
        sequence_output: $("sequenceOutput").value.trim(),
        sequence_left_pattern: $("sequenceLeftPattern").value.trim(),
        sequence_right_pattern: $("sequenceRightPattern").value.trim(),
        sequence_ext: $("sequenceExt").value,
        photo_mode: $("photoMode").value,
        photo_left: $("photoLeft").value.trim(),
        photo_right: $("photoRight").value.trim(),
        photo_sbs: $("photoSbsPath").value.trim(),
        photo_output: $("photoOutput").value.trim(),
        processing_backend: $("processingBackend").value,
        include_audio: $("includeAudio").checked,
        audio_source: $("audioSource").value,
        convergence: getConvergence(),
        vertical_offset: Number($("verticalOffset").value || 0),
        ghost_reduction: Number($("ghostReduction").value || 0),
        saturation: Number($("saturation").value || 100),
        contrast: Number($("contrast").value || 100),
        brightness: Number($("brightness").value || 0),
        left_gain: Number($("leftGain").value || 100),
        right_gain: Number($("rightGain").value || 100),
        red_gain: Number($("redGain").value || 100),
        cyan_gain: Number($("cyanGain").value || 100),
        linearize_srgb: $("linearizeSrgb").checked,
        swap_eyes: $("swapEyes").checked,
        trim_edges: $("trimEdges").checked,
        preview_percent: activeTab === "video" && videoDuration > 0
          ? Number($("previewPercent").value || 0) * 100 / videoDuration
          : Number($("previewPercent").value || 0),
        preview_seconds: Number($("previewPercent").value || 0)
      };
    }

    function showPlaceholder(message) {
      $("placeholder").textContent = message;
      $("placeholder").style.display = "block";
      $("previewImage").style.display = "none";
    }

    async function apiJson(path, data) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data || {})
      });
      const result = await response.json();
      if (!response.ok || result.ok === false) {
        throw new Error(result.error || "请求失败");
      }
      return result;
    }

    function schedulePreview() {
      clearTimeout(previewTimer);
      previewTimer = setTimeout(refreshPreview, 220);
    }

    async function refreshPreview() {
      try {
        const response = await fetch("/api/preview", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({task: activeTab, settings: collectSettings()})
        });
        if (!response.ok) {
          const error = await response.json().catch(() => ({error: "预览失败"}));
          showPlaceholder(error.error || "预览失败");
          return;
        }
        const blob = await response.blob();
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = URL.createObjectURL(blob);
        $("previewImage").src = previewUrl;
        $("previewImage").style.display = "block";
        $("placeholder").style.display = "none";
      } catch (error) {
        showPlaceholder(error.message);
      }
    }

    async function pick(kind, target) {
      try {
        $("statusBadge").textContent = "等待文件选择";
        const result = await apiJson("/api/pick", {kind});
        if (result.path) {
          $(target).value = result.path;
          if (["videoLeft", "videoRight", "videoSbsPath"].includes(target)) {
            refreshMediaInfo();
          }
          schedulePreview();
        }
      } catch (error) {
        alert(error.message);
      } finally {
        $("statusBadge").textContent = "本地运行 · OpenCV 处理";
      }
    }

    async function startTask(task) {
      try {
        const result = await apiJson("/api/start", {task, settings: collectSettings()});
        $("progressText").textContent = result.message;
        pollStatus();
      } catch (error) {
        alert(error.message);
      }
    }

    async function scanSequence() {
      try {
        const result = await apiJson("/api/sequence-count", {settings: collectSettings()});
        $("progressText").textContent = `找到 ${result.count} 组图片`;
        schedulePreview();
      } catch (error) {
        alert(error.message);
      }
    }

    async function refreshMediaInfo() {
      if (activeTab !== "video") return;
      try {
        const result = await apiJson("/api/media-info", {settings: collectSettings()});
        videoDuration = Number(result.duration || 0);
        $("previewPercent").max = videoDuration > 0 ? videoDuration : 100;
        $("previewPercent").step = videoDuration > 0 ? 0.1 : 1;
        $("previewPercentNum").max = $("previewPercent").max;
        $("previewPercentNum").step = $("previewPercent").step;
        if (Number($("previewPercent").value || 0) > Number($("previewPercent").max)) {
          $("previewPercent").value = $("previewPercent").max;
        }
        setReadouts();
      } catch (_error) {
        videoDuration = 0;
        setReadouts();
      }
    }

    async function refreshHardware() {
      try {
        const response = await fetch("/api/hardware");
        const hardware = await response.json();
        const gpuText = hardware.torch_cuda
          ? `CUDA：${hardware.torch_device}`
          : "CUDA：未检测到可用设备";
        const audioText = hardware.audio_mux ? "FFmpeg：可合并音频" : "FFmpeg：未找到，视频将无声";
        $("hardwareStatus").textContent = `${hardware.backend_label} · ${gpuText} · ${audioText}`;
      } catch (error) {
        $("hardwareStatus").textContent = error.message;
      }
    }

    async function openOutput(path) {
      try {
        await apiJson("/api/open-output", {path});
      } catch (error) {
        alert(error.message);
      }
    }

    async function searchPapers() {
      $("paperResults").textContent = "正在搜索论文...";
      try {
        const result = await apiJson("/api/papers", {});
        const suggestions = (result.suggestions || []).map((item) => `<div>${item}</div>`).join("");
        const papers = (result.results || result.curated || []).map((paper) => {
          const link = paper.url ? `<a href="${paper.url}" target="_blank">${paper.title}</a>` : paper.title;
          return `<div>${link} <span>(${paper.year || "n.d."})</span><br>${paper.note || ""}</div>`;
        }).join("");
        $("paperResults").innerHTML = `${suggestions}<hr>${papers || "没有搜索到结果，已保留内置建议。"}`;
      } catch (error) {
        $("paperResults").textContent = error.message;
      }
    }

    async function cancelTask() {
      await apiJson("/api/cancel", {});
      pollStatus();
    }

    async function pollStatus() {
      const response = await fetch("/api/status");
      const status = await response.json();
      $("progress").value = status.progress || 0;
      $("progressText").textContent = status.text || "准备就绪";
      $("log").textContent = (status.logs || []).join("\n");
      $("log").scrollTop = $("log").scrollHeight;
      $("statusBadge").textContent = status.running ? "正在转换" : "本地运行 · OpenCV 处理";
      if (status.running) {
        setTimeout(pollStatus, 600);
      }
    }

    function switchTab(tab) {
      activeTab = tab;
      document.querySelectorAll(".tab-button").forEach((button) => {
        button.classList.toggle("active", button.dataset.tab === tab);
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === `tab-${tab}`);
      });
      if (tab === "video") {
        refreshMediaInfo();
      } else {
        $("previewPercent").max = 100;
        $("previewPercent").step = 1;
        $("previewPercentNum").max = 100;
        $("previewPercentNum").step = 1;
        if (tab === "photo") {
          $("previewPercent").value = 0;
        }
        setReadouts();
      }
      schedulePreview();
    }

    function refreshModeGroups() {
      const videoSeparate = $("videoMode").value === "separate";
      $("videoSeparate").classList.toggle("active", videoSeparate);
      $("videoSbs").classList.toggle("active", !videoSeparate);

      const photoSeparate = $("photoMode").value === "photo_separate";
      $("photoSeparate").classList.toggle("active", photoSeparate);
      $("photoSbs").classList.toggle("active", !photoSeparate);
    }

    document.querySelectorAll(".tab-button").forEach((button) => {
      button.addEventListener("click", () => switchTab(button.dataset.tab));
    });
    document.querySelectorAll("[data-pick]").forEach((button) => {
      button.addEventListener("click", () => pick(button.dataset.pick, button.dataset.target));
    });
    document.querySelectorAll("input[type='text'], select").forEach((element) => {
      element.addEventListener("change", schedulePreview);
    });

    $("videoMode").addEventListener("change", () => { refreshModeGroups(); refreshMediaInfo(); });
    ["videoLeft", "videoRight", "videoSbsPath"].forEach((id) => {
      $(id).addEventListener("change", refreshMediaInfo);
    });
    $("photoMode").addEventListener("change", refreshModeGroups);
    $("trimEdges").addEventListener("change", schedulePreview);
    $("linearizeSrgb").addEventListener("change", schedulePreview);
    $("swapEyes").addEventListener("change", schedulePreview);
    $("includeAudio").addEventListener("change", schedulePreview);
    $("processingBackend").addEventListener("change", schedulePreview);
    $("audioSource").addEventListener("change", schedulePreview);
    $("previewVideo").addEventListener("click", refreshPreview);
    $("previewSequence").addEventListener("click", refreshPreview);
    $("previewPhoto").addEventListener("click", refreshPreview);
    $("startVideo").addEventListener("click", () => startTask("video"));
    $("startSequence").addEventListener("click", () => startTask("sequence"));
    $("startPhoto").addEventListener("click", () => startTask("photo"));
    $("scanSequence").addEventListener("click", scanSequence);
    $("cancelTask").addEventListener("click", cancelTask);
    $("openVideoOutput").addEventListener("click", () => openOutput($("videoOutput").value.trim()));
    $("openSequenceOutput").addEventListener("click", () => openOutput($("sequenceOutput").value.trim()));
    $("openPhotoOutput").addEventListener("click", () => openOutput($("photoOutput").value.trim()));
    $("refreshHardware").addEventListener("click", refreshHardware);
    $("searchPapers").addEventListener("click", searchPapers);

    syncPair("coarse", "coarseNum", 0);
    syncPair("fine", "fineNum", 1);
    syncPair("previewPercent", "previewPercentNum", 0);
    syncPair("verticalOffset", "verticalOffsetNum", 1);
    syncPair("ghostReduction", "ghostReductionNum", 0);
    syncPair("saturation", "saturationNum", 0);
    syncPair("contrast", "contrastNum", 0);
    syncPair("brightness", "brightnessNum", 0);
    syncPair("leftGain", "leftGainNum", 0);
    syncPair("rightGain", "rightGainNum", 0);
    syncPair("redGain", "redGainNum", 0);
    syncPair("cyanGain", "cyanGainNum", 0);
    setReadouts();
    refreshModeGroups();
    refreshHardware();
    refreshMediaInfo();
    pollStatus();
  </script>
</body>
</html>
"""


class AppHandler(BaseHTTPRequestHandler):
    server_version = "DuboisAnaglyph/1.0"

    def log_message(self, _format, *_args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            self.send_json(JOB.snapshot())
            return
        if parsed.path == "/api/hardware":
            self.send_json(detect_hardware())
            return
        self.send_error_json("未找到页面。", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/preview":
                self.handle_preview()
                return
            if parsed.path == "/api/pick":
                self.handle_pick()
                return
            if parsed.path == "/api/start":
                self.handle_start()
                return
            if parsed.path == "/api/cancel":
                JOB.cancel_event.set()
                self.send_json({"ok": True, "message": "正在停止当前任务..."})
                return
            if parsed.path == "/api/sequence-count":
                self.handle_sequence_count()
                return
            if parsed.path == "/api/media-info":
                self.handle_media_info()
                return
            if parsed.path == "/api/open-output":
                self.handle_open_output()
                return
            if parsed.path == "/api/papers":
                self.handle_papers()
                return
            self.send_error_json("未知接口。", 404)
        except Exception as exc:
            self.send_error_json(str(exc), 400)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        data = self.rfile.read(length)
        return json.loads(data.decode("utf-8"))

    def send_bytes(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_bytes(data, "application/json; charset=utf-8", status=status)

    def send_error_json(self, message, status=400):
        self.send_json({"ok": False, "error": message}, status=status)

    def handle_preview(self):
        payload = self.read_json()
        task = payload.get("task")
        settings = payload.get("settings") or {}
        left, right = load_preview_pair(task, settings)
        image = process_anaglyph(left, right, settings)
        image = fit_preview(image)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise ValueError("无法生成预览图。")
        self.send_bytes(encoded.tobytes(), "image/jpeg")

    def handle_pick(self):
        payload = self.read_json()
        path = pick_path(payload.get("kind"))
        self.send_json({"ok": True, "path": path})

    def handle_start(self):
        payload = self.read_json()
        task = payload.get("task")
        settings = payload.get("settings") or {}
        target = validate_start_payload(task, settings)
        JOB.start(target, settings)
        self.send_json({"ok": True, "message": "转换任务已开始。"})

    def handle_sequence_count(self):
        payload = self.read_json()
        settings = payload.get("settings") or {}
        pairs = discover_sequence_pairs(
            settings.get("sequence_folder") or "",
            settings.get("sequence_left_pattern") or "left_*.*",
            settings.get("sequence_right_pattern") or "right_*.*",
        )
        if not pairs:
            raise ValueError("没有找到可配对的左右眼图片。")
        self.send_json({"ok": True, "count": len(pairs)})

    def handle_media_info(self):
        payload = self.read_json()
        settings = payload.get("settings") or {}
        self.send_json(inspect_video_settings(settings))

    def handle_open_output(self):
        payload = self.read_json()
        path = payload.get("path")
        folder = open_output_location(path)
        self.send_json({"ok": True, "folder": folder})

    def handle_papers(self):
        self.send_json(search_papers())


def free_port(preferred):
    if preferred:
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def safe_print(message):
    try:
        print(message, flush=True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Dubois red/cyan anaglyph converter")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    port = free_port(args.port)
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    url = f"http://127.0.0.1:{port}/"
    safe_print(f"Dubois 红青立体转换器已启动：{url}")
    safe_print("关闭这个窗口即可退出程序。")
    if not args.no_browser and os.environ.get("DUBOIS_NO_BROWSER") != "1":
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
