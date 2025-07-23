import os
import glob
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from RRDBNet_arch import RRDBNet

# ================================
# CONFIGURATION
# ================================
model_path = './models/interp_08.pth'   # interpolated model or other .pth
test_img_folder = './LR/*'
results_folder = './results'
tile_size = 128       # adjust for memory (smaller -> less RAM)
tile_pad = 10          # overlap between tiles
scale = 4             # upscaling factor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(results_folder, exist_ok=True)

# ================================
# LOAD MODEL
# ================================
print(f"[INFO] Using device: {device}")
model = RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# ================================
# TILE INFERENCE FUNCTION
# ================================
def tile_process(img_LR):
    b, c, h, w = img_LR.size()
    h_out, w_out = h * scale, w * scale
    output = torch.zeros((c, h_out, w_out), device='cpu')

    # compute tiles
    stride = tile_size - tile_pad * 2
    h_tiles = (h - 1) // stride + 1
    w_tiles = (w - 1) // stride + 1

    for y in tqdm(range(h_tiles), desc="Processing tiles"):
        for x in range(w_tiles):
            y0 = y * stride
            x0 = x * stride
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)

            y0_pad = max(y0 - tile_pad, 0)
            x0_pad = max(x0 - tile_pad, 0)
            y1_pad = min(y1 + tile_pad, h)
            x1_pad = min(x1 + tile_pad, w)

            tile = img_LR[:, :, y0_pad:y1_pad, x0_pad:x1_pad].to(device)

            with torch.no_grad():
                sr_tile = model(tile).data.squeeze().float().cpu().clamp_(0, 1)

            # crop to remove pad and fit
            ty0 = (y0 - y0_pad) * scale
            tx0 = (x0 - x0_pad) * scale
            ty1 = ty0 + (y1 - y0) * scale
            tx1 = tx0 + (x1 - x0) * scale

            output[:, y0*scale:y1*scale, x0*scale:x1*scale] = sr_tile[:, ty0:ty1, tx0:tx1]

    return output


# ================================
# PROCESS IMAGES
# ================================
img_paths = glob.glob(test_img_folder)
if not img_paths:
    print("[ERROR] No images found in", test_img_folder)
    exit(1)

for idx, path in enumerate(img_paths):
    base = osp.splitext(osp.basename(path))[0]
    print(f"[INFO] Processing {idx+1}/{len(img_paths)}: {base}")

    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).unsqueeze(0)

    # upscale
    output = tile_process(img)

    # save
    output_img = np.transpose(output.numpy(), (1, 2, 0)) * 255.0
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)[:, :, ::-1]  # BGR
    save_path = osp.join(results_folder, f"{base}_upscaled.png")
    cv2.imwrite(save_path, output_img)
    print(f"[INFO] Saved: {save_path}")
