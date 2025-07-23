import sys
import torch
from collections import OrderedDict
from tqdm import tqdm
import logging

# --- logging setup ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- arguments ---
if len(sys.argv) < 2:
    logging.error("Usage: python interpolate.py <alpha>")
    sys.exit(1)

alpha = float(sys.argv[1])
assert 0.0 <= alpha <= 1.0, "alpha must be between 0 and 1"

net_PSNR_path = './models/RRDB_PSNR_x4.pth'
net_ESRGAN_path = './models/RRDB_ESRGAN_x4.pth'
net_interp_path = './models/interp_{:02d}.pth'.format(int(alpha * 10))

# --- device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# --- load models ---
logging.info("Loading PSNR model...")
net_PSNR = torch.load(net_PSNR_path, map_location=device)

logging.info("Loading ESRGAN model...")
net_ESRGAN = torch.load(net_ESRGAN_path, map_location=device)

net_interp = OrderedDict()

logging.info(f"Interpolating weights with alpha = {alpha:.2f}")

# --- interpolation ---
for k in tqdm(net_PSNR.keys(), desc="Interpolating weights"):
    v_PSNR = net_PSNR[k].to(device)
    v_ESRGAN = net_ESRGAN[k].to(device)
    net_interp[k] = ((1 - alpha) * v_PSNR + alpha * v_ESRGAN).cpu()

# --- save ---
torch.save(net_interp, net_interp_path)
logging.info(f"Saved interpolated model to: {net_interp_path}")
