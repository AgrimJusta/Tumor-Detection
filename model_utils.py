# model_utils.py
import os
import time
import numpy as np
from PIL import Image
import io

# Try to import torch; if not available we fallback to a simulator
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Try to import convolve2d from scipy.signal; if not available provide a numpy fallback
try:
    from scipy.signal import convolve2d
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

def numpy_convolve2d(a, kernel, mode='same'):
    """
    Minimal 2D convolution fallback using numpy (valid, same modes supported).
    This is a simple implementation and not optimized for large kernels.
    """
    a = np.asarray(a, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    kh, kw = kernel.shape
    # flip kernel for convolution
    kernel_flipped = kernel[::-1, ::-1]

    if mode == 'same':
        pad_h = kh // 2
        pad_w = kw // 2
        padded = np.pad(a, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    elif mode == 'valid':
        padded = a
    else:
        # fallback to same if unknown
        pad_h = kh // 2
        pad_w = kw // 2
        padded = np.pad(a, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    out_h = a.shape[0]
    out_w = a.shape[1]
    out = np.zeros((out_h, out_w), dtype=float)

    for i in range(out_h):
        for j in range(out_w):
            patch = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(patch * kernel_flipped)
    return out

def box_blur(img_gray, kernel_size=15):
    """
    Blur the grayscale image with a uniform kernel (box blur).
    Uses scipy.convolve2d if available, else numpy fallback.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size * kernel_size)
    if SCIPY_AVAILABLE:
        return convolve2d(img_gray, kernel, mode='same', boundary='symm')
    else:
        return numpy_convolve2d(img_gray, kernel, mode='same')

def get_model(num_classes=2, pretrained=False):
    """
    Return a model object. If torch is available, return a small torchvision resnet.
    Otherwise return a dummy object for simulation.
    """
    if TORCH_AVAILABLE:
        model = models.resnet18(pretrained=pretrained)
        # replace final layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        # simple dummy model object
        class Dummy:
            def eval(self): pass
            def __call__(self, x):
                # return random logits-like array
                logits = np.random.rand(1, 2)
                return logits
        return Dummy()

def load_image_for_infer(path, target_size=(224,224)):
    """
    Return an image tensor or numpy array suitable for infer_image.
    If torch is available it returns a torch tensor (C,H,W) normalized;
    otherwise returns a numpy array HxWxC normalized to [0,1].
    """
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0

    if TORCH_AVAILABLE:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        return preprocess(img)  # returns torch.Tensor CxHxW
    else:
        # return HxWxC
        return arr

def generate_heatmap_overlay(rgb_img_arr, intensity_map, save_path):
    """
    Create a heatmap overlay from a base RGB numpy array and an intensity_map (0..1 float array).
    - rgb_img_arr: HxWx3 in 0..1
    - intensity_map: HxW in 0..1
    Saves the overlay to save_path (PNG).
    """
    from matplotlib import cm

    # Ensure intensity_map is 0..1
    imap = np.clip(intensity_map, 0.0, 1.0)

    # Resize intensity map to image if shapes mismatch
    if imap.shape != rgb_img_arr.shape[:2]:
        # simple resize using PIL
        im_pil = Image.fromarray((imap * 255).astype(np.uint8))
        im_pil = im_pil.resize((rgb_img_arr.shape[1], rgb_img_arr.shape[0]), resample=Image.BILINEAR)
        imap = np.array(im_pil).astype(np.float32) / 255.0

    base = (rgb_img_arr * 255).astype(np.uint8)
    cmap = cm.get_cmap('jet')
    heatmap_rgb = (cmap(imap)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.55 * base + 0.45 * heatmap_rgb).astype(np.uint8)

    # ensure directory exists
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    Image.fromarray(overlay).save(save_path)
    return save_path

def infer_image(model, image_path, model_path=None, return_heatmap=False):
    """
    Runs inference and optionally generates a heatmap overlay.
    Returns (probs_array, heatmap_path_or_None)
    - If torch available and model_path provided, loads model weights (cpu)
    """
    img_tensor = load_image_for_infer(image_path)  # either torch.Tensor CxHxW or numpy HxWxC

    # load model weights if possible
    if TORCH_AVAILABLE and model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        except Exception:
            # ignore load failure (maybe incompatible)
            pass

    # run forward & softmax
    if TORCH_AVAILABLE:
        model.eval()
        with torch.no_grad():
            if isinstance(img_tensor, torch.Tensor):
                inp = img_tensor.unsqueeze(0)  # 1xCxHxW
            else:
                inp = torch.tensor(img_tensor).permute(2,0,1).unsqueeze(0)
            logits = model(inp)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    else:
        # fallback simulated prediction
        probs = np.random.rand(2)
        probs = probs / probs.sum()

    heatmap_path = None
    if return_heatmap:
        # Prepare rgb arr in 0..1
        if TORCH_AVAILABLE:
            if isinstance(img_tensor, np.ndarray):
                img_arr = img_tensor
            else:
                try:
                    t = img_tensor.clone()
                    # try to approximate unnormalize (if normalized)
                    mean = np.array([0.485,0.456,0.406]).reshape(3,1,1)
                    std = np.array([0.229,0.224,0.225]).reshape(3,1,1)
                    t = t * torch.tensor(std) + torch.tensor(mean)
                    img_arr = t.permute(1,2,0).cpu().numpy()
                except Exception:
                    # last resort convert via PIL
                    img_pil = Image.open(image_path).convert('RGB').resize((224,224))
                    img_arr = np.array(img_pil).astype(np.float32) / 255.0
        else:
            img_arr = img_tensor  # already 0..1

        # Build a simple intensity map from luminance and blur it
        gray = np.dot(img_arr[...,:3], [0.299, 0.587, 0.114])
        try:
            blurred = box_blur(gray, kernel_size=15)
        except Exception:
            blurred = gray

        intensity = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-9)
        heatmap_name = "heatmap_" + os.path.basename(image_path).rsplit('.',1)[0] + ".png"
        heatmap_path = os.path.join("uploads", heatmap_name)
        os.makedirs(os.path.dirname(heatmap_path) or '.', exist_ok=True)
        generate_heatmap_overlay(img_arr, intensity, heatmap_path)

    return probs, heatmap_path

def train_from_folder(dataset_folder, model_save_path, epochs=3, lr=1e-3, status_callback=None):
    """
    Simulated training helper that writes a model file and calls status_callback with updates.
    If torch is available you can (optionally) implement real training here. For demo, we simulate.
    status_callback should accept a dict with keys like 'phase','progress','last_message'
    """
    total_steps = epochs * 10
    step = 0
    if status_callback:
        status_callback({'phase':'training','progress':0,'last_message':'Starting training...'})

    for e in range(1, epochs+1):
        for i in range(10):
            time.sleep(0.4)  # simulate work
            step += 1
            prog = int((step / total_steps) * 100)
            msg = f"Epoch {e}/{epochs} step {i+1}/10"
            if status_callback:
                status_callback({'phase':'training','progress':prog,'last_message':msg})
        # simulate saving checkpoint each epoch
        with open(model_save_path, 'ab') as f:
            f.write(f"epoch-{e}\n".encode('utf-8'))
    # final save
    if status_callback:
        status_callback({'phase':'finishing','progress':100,'last_message':'Finalizing and saving model...'})

    return model_save_path
