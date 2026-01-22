# 🎥 AI Video Enhancer & Upscaler

A **local-first, privacy-friendly AI video upscaling tool** built in Python.  
Upscale videos to **1080p and 4K** using **Real-ESRGAN (AI)** with automatic fallback to **FSRCNN (CPU)**.

No cloud. No uploads. Fully offline.

---

## ✨ Features

### AI Upscaling
- Real-ESRGAN for high-quality AI enhancement  
- NVIDIA CUDA support  
- Apple Silicon (MPS) support  

### Smart Fallback
- Automatically switches to OpenCV **FSRCNN**  
- Works on CPU-only systems without configuration  

### 1080p → 4K Pipeline
- AI upscale to 1080p  
- High-quality **Lanczos interpolation** for 4K delivery  

### Aspect Ratio Handling
- Automatic **4:3 → 16:9** correction  
- Smart padding (no stretching or cropping)  

### Audio Preservation
- Extracts original audio  
- Lossless remux into upscaled video  

### Local & Offline
- No uploads  
- No server-side processing  
- No data leaves your machine  

---

## 🖥 Platform & Hardware Support

### Operating Systems
- macOS (Intel & Apple Silicon)  
- Linux  
- Windows  

### Acceleration Backends

| Hardware | Backend | Status |
|--------|--------|--------|
| NVIDIA GPU | CUDA | Real-ESRGAN |
| Apple Silicon (M1–M3) | MPS | Real-ESRGAN |
| CPU-only | OpenCV FSRCNN | Fallback |

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+**  
- **FFmpeg** (must be available in PATH)  

---

### Install FFmpeg

#### macOS
```bash
brew install ffmpeg
```

#### Linux
```bash
sudo apt install ffmpeg
```

#### Windows
Download from: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)  
Ensure `ffmpeg` is added to PATH.

---

## 📦 Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/pratik227/video-enhance.git
   cd video-enhance
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights**

   **AI Upscaling (Recommended)**  
   Place one of the following files in the project root:
   - `RealESRGAN_x4plus.pth`
   - `RealESRGAN_x4.pth`

   **CPU Fallback**  
   Place this file in the `models/` directory:
   - `FSRCNN_x3.pb`

---

## 🛠 Usage
Rename your input video to match the default name OR update the `INPUT_VIDEO` variable in the script.

```bash
python upscale_video.py
```

### Output
- AI-enhanced 1080p video
- High-quality 4K video
- Original audio preserved

---

## 📚 Python Dependencies
- `torch`
- `opencv-python`
- `ffmpeg-python`
- `numpy`
- `tqdm`
- `realesrgan`
- `ai-forever Real-ESRGAN`

---

## ⚙️ Processing Notes
- GPU acceleration is auto-detected at runtime
- FSRCNN fallback is used automatically if GPU is unavailable
- Optimized for long-form and archival video processing

### Suitable for
- Old SD footage
- Personal media restoration
- Content remastering
- Offline processing pipelines

---

## 💖 Support & Sponsorship
If this project helps you, consider supporting its development.

💝 **GitHub Sponsors**  
[https://github.com/sponsors/pratik227](https://github.com/sponsors/pratik227)

☕ **Buy Me a Coffee**  
[https://buymeacoffee.com/pratik227](https://buymeacoffee.com/pratik227)

⭐ **Free Support**  
- Star the repository
- Share on X / Reddit / Hacker News
- Recommend it to others

---

## 🧑‍💻 Maintainer
**Pratik Patel**  
Independent builder focused on local-first, privacy-respecting tools

GitHub: [https://github.com/pratik227](https://github.com/pratik227)

---

## 📄 License
MIT License  
See the [LICENSE](LICENSE) file for details.
