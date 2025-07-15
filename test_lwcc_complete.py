import cv2, matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil

# Create local cache directory
local_cache = os.path.expanduser("~/lwcc_cache")
local_weights = os.path.join(local_cache, "weights")
os.makedirs(local_weights, exist_ok=True)

# Patch key functions
original_mkdir = Path.mkdir
original_listdir = os.listdir
original_move = shutil.move

def redirect_path(path):
    """Redirect /.lwcc paths to local cache"""
    path_str = str(path)
    if path_str.startswith("/.lwcc"):
        return path_str.replace("/.lwcc", local_cache)
    return path_str

def patched_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    redirected = redirect_path(self)
    if redirected != str(self):
        redirected_path = Path(redirected)
        return original_mkdir(redirected_path, mode, parents, exist_ok)
    return original_mkdir(self, mode, parents, exist_ok)

def patched_listdir(path):
    redirected = redirect_path(path)
    return original_listdir(redirected)

def patched_move(src, dst):
    redirected_dst = redirect_path(dst)
    return original_move(src, redirected_dst)

# Apply patches
Path.mkdir = patched_mkdir
os.listdir = patched_listdir
shutil.move = patched_move

# Patch os.path.dirname for gdown
import os.path as osp
original_dirname = osp.dirname

def patched_dirname(path):
    result = original_dirname(path)
    return redirect_path(result)

osp.dirname = patched_dirname

# Patch torch.load
import torch
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, **pickle_load_args):
    if isinstance(f, str):
        f = redirect_path(f)
    return original_torch_load(f, map_location, pickle_module, **pickle_load_args)

torch.load = patched_torch_load

from lwcc import LWCC

IMG_PATH = "test_image_2.jpg"

# pick a light model/weight combo – SHB is best for sparse (<50)
count, density = LWCC.get_count(
        IMG_PATH,
        model_name="CSRNet",
        model_weights="SHB",
        return_density=True
)

print(f"Predicted count ≈ {count:.1f}")

# visualise density map
heat = cv2.applyColorMap(
          (density / density.max() * 255).astype("uint8"),
          cv2.COLORMAP_JET)
heat = cv2.resize(heat, (density.shape[1]*2, density.shape[0]*2),
                  interpolation=cv2.INTER_CUBIC)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1);  plt.title("RGB");     plt.imshow(cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)); plt.axis('off')
plt.subplot(1,2,2);  plt.title("LCD density"); plt.imshow(heat);             plt.axis('off')
plt.tight_layout();   plt.show() 