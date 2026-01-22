import ffmpeg
import os

INPUT_1080P = "output_1080p.mp4"
OUTPUT_4K = "output_4k_delivery.mp4"

def create_4k_delivery():
    if not os.path.exists(INPUT_1080P):
        print(f"[ERROR] Input file {INPUT_1080P} not found.")
        return

    print(f"[INFO] Creating 4K delivery file from {INPUT_1080P}...")
    try:
        (
            ffmpeg
            .input(INPUT_1080P)
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
        print(f"[DONE] 4K Delivery complete: {OUTPUT_4K}")
    except ffmpeg.Error as e:
        print(f"[ERROR] FFmpeg failed: {e}")

if __name__ == "__main__":
    create_4k_delivery()
