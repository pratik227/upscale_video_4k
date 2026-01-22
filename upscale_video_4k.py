import os
import cv2
import torch
import ffmpeg
import numpy as np
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
INPUT_VIDEO = "assad).mp4"
OUTPUT_1080P = "output_1080p.mp4"
OUTPUT_4K = "output_4k_delivery.mp4"
TEMP_AUDIO = "temp_audio.aac"
CREATE_4K_DELIVERY = True

TARGET_W, TARGET_H = 1920, 1080
UPSCALE_FACTOR_GPU = 4
UPSCALE_FACTOR_CPU = 3

# ===============================
# HARDWARE DETECTION
# ===============================
if torch.cuda.is_available():
    device = torch.device("cuda")
    USE_GPU = True
    print("[INFO] NVIDIA GPU detected (CUDA)")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    USE_GPU = True
    print("[INFO] Apple Silicon GPU detected (MPS)")
else:
    device = torch.device("cpu")
    USE_GPU = False
    print("[INFO] Using CPU")

# ===============================
# LOAD UPSCALER
# ===============================
if USE_GPU:
    try:
        from realesrgan import RealESRGAN
    except ImportError:
        print("[ERROR] realesrgan package not found. Run: pip install realesrgan")
        exit(1)

    upscaler = RealESRGAN(device, scale=UPSCALE_FACTOR_GPU)
    # Note: load_weights might need a path or downloded automatically
    # check if path exists or let it download
    weights_path = "RealESRGAN_x4plus.pth"
    if not os.path.exists(weights_path):
        print(f"[INFO] {weights_path} not found. Ensure it is in the project root.")
    
    upscaler.load_weights(weights_path)
    print("[INFO] Using Real-ESRGAN x4")
else:
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    model_path = "models/FSRCNN_x3.pb"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("Please download it from: https://github.com/Saafke/FSRCNN_Tensorflow")
        exit(1)
    sr.readModel(model_path)
    sr.setModel("fsrcnn", UPSCALE_FACTOR_CPU)
    print("[INFO] Using OpenCV FSRCNN x3")

# ===============================
# FFmpeg: Extract Audio
# ===============================
if not os.path.exists(INPUT_VIDEO):
    print(f"[ERROR] Input video not found: {INPUT_VIDEO}")
    exit(1)

print("[INFO] Extracting audio...")
try:
    ffmpeg.input(INPUT_VIDEO).output(
        TEMP_AUDIO, acodec="copy", vn=None
    ).overwrite_output().run(quiet=True)
except ffmpeg.Error as e:
    print(f"[WARNING] Could not extract audio (maybe no audio track?): {e}")
    TEMP_AUDIO = None

# ===============================
# Video Metadata
# ===============================
probe = ffmpeg.probe(INPUT_VIDEO)
video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")

fps = eval(video_stream["r_frame_rate"])
width = int(video_stream["width"])
height = int(video_stream["height"])
total_frames = int(video_stream.get("nb_frames") or 0)

if total_frames == 0:
    # Estimate total frames if not in metadata
    duration = float(video_stream.get("duration", 0))
    if duration > 0:
        total_frames = int(duration * fps)

# ===============================
# FFmpeg OUTPUT PIPE
# ===============================
output_args = {
    'vcodec': 'libx264',
    'pix_fmt': 'yuv420p',
    'r': fps,
}

if TEMP_AUDIO and os.path.exists(TEMP_AUDIO):
    output_args['acodec'] = 'copy'
    input_audio = ffmpeg.input(TEMP_AUDIO)
    video_input = ffmpeg.input(
        "pipe:",
        format="rawvideo",
        pix_fmt="bgr24",
        s=f"{TARGET_W}x{TARGET_H}",
        r=fps
    )
    process = (
        ffmpeg
        .output(video_input, input_audio, OUTPUT_1080P, **output_args)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
else:
    process = (
        ffmpeg
        .input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{TARGET_W}x{TARGET_H}",
            r=fps
        )
        .output(OUTPUT_1080P, **output_args)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

# ===============================
# PROCESS VIDEO
# ===============================
cap = cv2.VideoCapture(INPUT_VIDEO)

with tqdm(total=total_frames, desc="Upscaling") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -----------------------
        # SUPER-RESOLUTION
        # -----------------------
        if USE_GPU:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sr_frame = upscaler.predict(frame_rgb)
            sr_frame = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
        else:
            sr_frame = sr.upsample(frame)

        # -----------------------
        # GEOMETRY FIX (4:3 → 16:9)
        # -----------------------
        h, w = sr_frame.shape[:2]
        scale = TARGET_H / h
        resized = cv2.resize(
            sr_frame,
            (int(w * scale), TARGET_H),
            interpolation=cv2.INTER_LANCZOS4
        )

        pad_total = TARGET_W - resized.shape[1]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        if pad_total > 0:
            final_frame = cv2.copyMakeBorder(
                resized,
                0, 0,
                pad_left, pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
        else:
            # Crop if slightly over (unlikely with scale = TARGET_H / h)
            final_frame = resized[:, :TARGET_W]

        # -----------------------
        # WRITE TO PIPE
        # -----------------------
        process.stdin.write(final_frame.astype(np.uint8).tobytes())
        pbar.update(1)

cap.release()
process.stdin.close()
process.wait()

# ===============================
# CLEANUP
# ===============================
if TEMP_AUDIO and os.path.exists(TEMP_AUDIO):
    os.remove(TEMP_AUDIO)

print("[DONE] 1080p Master complete:", OUTPUT_1080P)

# ===============================
# 4K DELIVERY UPSCALE
# ===============================
if CREATE_4K_DELIVERY:
    print(f"[INFO] Creating 4K delivery file (Lanczos upscale)...")
    try:
        (
            ffmpeg
            .input(OUTPUT_1080P)
            .filter("scale", 3840, 2160, flags="lanczos")
            .output(
                OUTPUT_4K,
                vcodec="libx264",
                acodec="copy",
                preset="slow",
                crf=16,
                pix_fmt="yuv420p"
            )
            .overwrite_output()
            .run(quiet=False)
        )
        print("[DONE] 4K Delivery complete:", OUTPUT_4K)
    except ffmpeg.Error as e:
        print(f"[ERROR] 4K upscale failed: {e}")
