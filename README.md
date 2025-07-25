# Mbisha Enhancer 🔬📸

**Mbisha Enhancer** is an advanced image upscaling and enhancement tool built using **Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN)**. It restores detail in low-resolution images and intelligently sharpens and denoises photos—perfect for photo restoration, social media cleanup, and more.

---

## ✨ Features

- ✅ **Super-resolution** using ESRGAN
- 🧠 **Interpolated models** between PSNR-oriented and perceptual weights
- 🧩 **Tile-based processing** to support low-memory environments (CPU-friendly)
- 🐍 Built in **Python** using **PyTorch**
- 💾 Save high-quality enhanced outputs
- 📊 Real-time **progress bars** during enhancement
- 🧪 Includes custom test pipeline with support for batch processing

---

## 📸 Before vs After

| Original                           | Enhanced                          |
| ---------------------------------- | --------------------------------- |
| ![before](./figures/sample_lr.png) | ![after](./figures/sample_sr.png) |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Eye-Kyu/Mbisha-Enhancer.git
cd Mbisha-Enhancer
```

### 2. Set up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Pretrained Models

Put your pre-trained ESRGAN `.pth` models in the `./models` directory. You can interpolate two models (e.g., PSNR and perceptual) like so:

```bash
python interpolate.py 0.8  # Alpha can be between 0.0 and 1.0
```

---

## 🧪 Running the Enhancer

### Single Image or Batch

Put your input images in the `./LR/` folder. Then run:

```bash
python test.py
```

Enhanced images will be saved in the `./results/` folder.

---

## ⚙️ Configuration

You can adjust tile size, padding, and model paths in `test.py`:

```python
tile_size = 256        # Size of each tile
tile_overlap = 16      # Overlap to reduce edge artifacts
model_path = './models/interp_08.pth'
```

---

## 📁 Project Structure

```
Mbisha-Enhancer/
├── LR/                    # Input low-res images
├── models/                # Pre-trained or interpolated ESRGAN models
├── results/               # Output folder for enhanced images
├── interpolate.py         # Script to blend PSNR and perceptual models
├── test.py                # Enhanced test script with tiling & progress bar
├── RRDBNet_arch.py        # ESRGAN model architecture
├── figures/               # Optional: visuals for README/demo
└── README.md              # You are here
```

---

## 🔧 Dependencies

- Python 3.7+
- PyTorch
- tqdm
- Pillow
- OpenCV

Install all with:

```bash
pip install -r requirements.txt
```

---

## 🧠 Inspiration

The inspiration for this project came about when trying to visually enhance a surveillance image of an intruder captured on a CCTV camera after other AI enhancement tools fell short. The ESRGAN architecture proved flexible and tuneable enough to achieve a clearer and more useable result from serverely degraded footage.

---

## 👤 Author

**Pancrass Muiruri**
Front-End Developer | Network Engineer | AI Hobbyist
📍 Nairobi, Kenya
🔗 [GitHub](https://github.com/Eye-Kyu) • [LinkedIn](https://www.linkedin.com/in/pancrass-muiruri)

---

## 📜 License

This project is open-source under the [MIT License](./LICENSE).
