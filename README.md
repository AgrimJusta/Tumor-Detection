# Tumor Detection — Local Interactive Demo

This repository is a local-first demo for a tumor-detection case study using CNNs (PyTorch) and a small Flask UI.
It is designed to run entirely on your machine and is intended for demonstration and learning purposes only.

## Features included
- Upload images (PNG/JPG) via web UI
- Start a demo training loop (simulated or configurable real training using ImageFolder)
- Poll training status and display simple progress
- Single-image inference endpoint (returns prediction probability)
- Simple Grad-CAM helper and model utilities
- Save/load model files under `models/`
- Exported ZIP with project for local run

## Quick start (CPU-only)
1. Clone the repository
   If you dont know how to clone [Click Here](https://youtu.be/q9wc7hUrW8U?si=h9WQRkWH4fayA4JN)   

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Prepare dataset:
   - Put images arranged as `uploads/dataset/train/class_name/*.jpg` and `uploads/dataset/val/class_name/*.jpg`
   - Or use the Upload page to add images manually.
5. Run the app:
   ```bash
   python app.py
   ```
6. Open browser: http://127.0.0.1:5000
   If the port is busy try running 
   ```bash
   python -c "from app import app; app.run(host='127.0.0.1', port=5001, debug=True)"
   ```
   you can replace the port with any available ones.

## Notes
- The included training loop in `model_utils.py` is small and intended as a starting point. Adapt dataset paths, transforms, and model saving for your use.
- For DICOM support, the `pydicom` package is included; uploaded DICOM files are converted to PNG for display.
- If you have a GPU and compatible torch build, adjust device selection in the code to use CUDA.

## Files
- `app.py` — Flask app with upload, start training, status, and infer routes.
- `model_utils.py` — PyTorch model, training loop, inference, and Grad-CAM helper.
- `templates/` — Basic HTML templates for the demo UI.
- `requirements.txt` — Python deps.
- `README.md` — This file.

Enjoy — ask me if you want the app extended with full training for your dataset, prettier UI, or Dockerfile.
