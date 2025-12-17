import os
import glob
from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from typing import Tuple
import tqdm

#!/usr/bin/env python3
"""
infer_segmentation.py

Minimal inference script for the DeepLabV3 segmentation model used in training script.
Saves predicted binary masks (0/1) as PNGs and optional overlay images.
"""

import torchvision.transforms as T

# -------------------------
# Dataset for inference
# -------------------------
class InferenceDataset(Dataset):
    """
    Loads images for inference. Returns (tensor_image, original_image_path).
    """
    def __init__(self, images_dir: str, image_size: Tuple[int, int] = (320, 320)):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize(self.image_size, interpolation=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert("L")    # original code was grayscale -> convert to RGB
        img = img.convert("RGB")
        img_t = self.transform(img)
        return img_t, path

# -------------------------
# Model helper (matches training)
# -------------------------
def create_model(pretrained=False, pretrained_backbone=False):
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=True)
    # replace final classifier conv to match 2 classes
    # classifier is e.g. DeepLabHead(2048, classifier=...)
    # previous training code used model.classifier[-1] = Conv2d(256, 2, 1)
    # keep same replacement to ensure compatibility
    model.classifier[-1] = torch.nn.Conv2d(256, 2, kernel_size=1)
    if pretrained_backbone:
        backbone = torchvision.models.resnet50(pretrained=True)
        model.backbone.load_state_dict(
            {k: v for k, v in backbone.state_dict().items() if k in model.backbone.state_dict()},
            strict=False
        )
    return model

# -------------------------
# Utilities
# -------------------------
def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    ck = torch.load(checkpoint_path, map_location=device)
    # training saved dict with 'model_state' or raw state_dict
    if isinstance(ck, dict) and 'model_state' in ck:
        state = ck['model_state']
    elif isinstance(ck, dict) and 'state_dict' in ck:
        state = ck['state_dict']
    else:
        state = ck
    model.load_state_dict(state)
    return model

def save_mask(mask_np: np.ndarray, out_path: str):
    # mask_np expected shape (H,W) with integer class ids {0,1}
    # Save as 8-bit PNG (0 or 255) for visibility
    im = Image.fromarray((mask_np.astype(np.uint8) * 255))
    im.save(out_path)

def save_overlay(resized_rgb: Image.Image, mask_np: np.ndarray, out_path: str, color=(255,0,0), alpha=0.5):
    """
    Create a simple overlay: colored mask blended over the resized RGB image.
    mask_np values 0/1.
    """
    base = resized_rgb.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    # color map
    mask_img = Image.fromarray((mask_np.astype(np.uint8) * 255))
    color_img = Image.new("RGBA", base.size, color + (0,))
    # create colored mask where mask==1 set alpha channel
    mask_alpha = mask_img.point(lambda p: int(p * alpha))
    color_img.putalpha(mask_alpha)
    overlay = Image.alpha_composite(base, color_img)
    overlay.convert("RGB").save(out_path)

# -------------------------
# Inference
# -------------------------
def run_inference(
    model_path: str,
    input_dir: str,
    output_dir: str,
    image_size=(320,320),
    batch_size: int = 4,
    device: str = None,
):
    device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    os.makedirs(output_dir, exist_ok=True)
    masks_out_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_out_dir, exist_ok=True)

    ds = InferenceDataset(input_dir, image_size=image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = create_model(pretrained=False).to(device)
    model.eval()
    model = load_checkpoint(model, model_path, device)
    model.to(device)
    with torch.no_grad():
        for imgs, paths in tqdm.tqdm(loader):
            imgs = imgs.to(device)
            out = model(imgs)['out']  # (N,C,H,W)
            preds = out.argmax(dim=1).cpu().numpy().astype(np.uint8)  # (N,H,W)
            for i, p in enumerate(paths):
                fname = os.path.splitext(os.path.basename(p))[0]
                mask_np = preds[i]
                mask_path = os.path.join(masks_out_dir, f"{fname}_mask.png")
                save_mask(mask_np, mask_path)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Path to saved model (.pth)")
    p.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
    p.add_argument("--output_dir", type=str, default="inference_outputs", help="Where to save predicted masks and overlays")
    p.add_argument("--image_size", type=int, nargs=2, default=(320,320), help="Resize input to this size (W H)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", type=str, default=None, help="torch device string, e.g. 'cpu' or 'cuda:0'. Default auto")
    args = p.parse_args()

    run_inference(
        model_path=args.model_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        device=args.device
    )