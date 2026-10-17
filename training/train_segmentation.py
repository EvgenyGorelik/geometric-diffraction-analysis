import os
import glob
from PIL import Image
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, optim
from tqdm import tqdm
import torchvision.transforms as T
from argparse import ArgumentParser

# -------------------------
# Dataset
# -------------------------
class SegmentationDataset(Dataset):
    """
    Expects:
      - images_dir: files like /path/to/images/img_001.jpg
      - masks_dir:  files like /path/to/masks/img_001.png
    Masks should contain integer class ids per pixel (0..num_classes-1).
    """
    def __init__(self, images_dir, masks_dir, image_size=(320, 320), augment=False):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*")))
        self.masks = sorted(glob.glob(os.path.join(masks_dir, "*")))
        assert len(self.images) == len(self.masks), "Mismatched number of images and masks"
        self.image_size = image_size
        self.augment = augment

        # image transform
        self.img_transform = T.Compose([
            T.Resize(self.image_size, interpolation=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("L")
        img = img.convert("RGB")  # convert grayscale to 3-channel RGB
        mask = Image.open(self.masks[idx]).convert("L")  # single-channel with class ids
        # Binarize mask: pixels >= 128 -> 1, else 0
        threshold = 128
        mask = mask.point(lambda p: 1 if p >= threshold else 0)

        # Optional simple augmentation (flip)
        if self.augment and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img = self.img_transform(img)
        # resize mask with nearest to keep integer labels
        mask = mask.resize(self.image_size, resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64)).long()  # assuming 255 is ignore index
        return img, mask

# -------------------------
# Metrics
# -------------------------
def compute_iou(pred, target, ignore_index=None):
    # pred: (N,H,W) int, target: (N,H,W) int
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(2):
        if ignore_index is not None and cls == ignore_index:
            continue
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))  # class not present in batch
        else:
            ious.append(intersection / union)
    # return mean IoU ignoring NaNs
    ious = [v for v in ious if not np.isnan(v)]
    return float(np.mean(ious)) if ious else 0.0

# -------------------------
# Model creation helper
# -------------------------
def create_model(pretrained=False, pretrained_backbone=False, checkpoint_path=None):
    # Using DeepLabV3 with ResNet50 backbone; change to other models if desired
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=True)
    # replace classifier to match num_classes
    model.classifier[-1] = torch.nn.Conv2d(256, 2, kernel_size=1)
    # optionally load pretrained backbone weights
    if pretrained_backbone:
        # load backbone pretrained weights only (example approach)
        backbone = torchvision.models.resnet50(pretrained=True)
        model.backbone.load_state_dict({k.replace('layer1.', 'layer1.'): v for k, v in backbone.state_dict().items() if k in model.backbone.state_dict()}, strict=False)

    if checkpoint_path is not None:
        # Load model weights from checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model_state'])

    return model


def create_class_balanced_dataloader(dataset, batch_size, zonal_labels, oversampling_ratio=0.8):
    assert 0.0 <= oversampling_ratio <= 1.0, "oversampling_ratio should be in (0,1)"
    # list of basenames for quick membership checks
    mask_basenames = [os.path.basename(p) for p in dataset.masks]
    zonal_set = set(zonal_labels)

    # per-sample weights: higher for zonal samples
    weights = [oversampling_ratio if name in zonal_set else (1.0 - oversampling_ratio) for name in mask_basenames]
    weights_tensor = torch.as_tensor(weights, dtype=torch.double)

    # Weighted sampler (replacement=True to allow oversampling)
    sampler = torch.utils.data.WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)

    # DataLoader using the sampler
    imbalanced_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    return imbalanced_loader

# -------------------------
# Training loop
# -------------------------
def train_segmentation(
    train_images_dir,
    train_masks_dir,
    val_images_dir=None,
    val_masks_dir=None,
    pretrained=False,
    image_size=(320,320),
    batch_size=8,
    lr=1e-4,
    weight_decay=1e-5,
    num_epochs=20,
    oversampling_ratio=0.8,
    device=None,
    save_path="seg_model.pth",
    zonal_labels=None,
    save_strategy="best_val",
    checkpoint_path=None
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    train_ds = SegmentationDataset(train_images_dir, train_masks_dir, image_size=image_size, augment=False)
    if zonal_labels is None:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    else:
        train_loader = create_class_balanced_dataloader(train_ds, batch_size, zonal_labels, oversampling_ratio=oversampling_ratio)

    val_loader = None
    if val_images_dir and val_masks_dir:
        val_ds = SegmentationDataset(val_images_dir, val_masks_dir, image_size=image_size, augment=False)
        if zonal_labels is None:
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        else:
            val_loader = create_class_balanced_dataloader(val_ds, batch_size, zonal_labels, oversampling_ratio=oversampling_ratio)


    model = create_model(pretrained=pretrained, checkpoint_path=checkpoint_path).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    val_iou_best = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)  # shape (N,H,W), dtype long
            optimizer.zero_grad()
            out = model(imgs)['out']  # (N,C,H,W)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        msg = f"Epoch {epoch}/{num_epochs} - train loss: {epoch_loss:.4f}"

        if val_loader:
            model.eval()
            with torch.no_grad():
                total_iou = 0.0
                n_batches = 0
                for imgs, masks in val_loader:
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    out = model(imgs)['out']
                    preds = out.argmax(dim=1)  # (N,H,W)
                    total_iou += compute_iou(preds.cpu(), masks.cpu())
                    n_batches += 1
                mean_iou = total_iou / n_batches if n_batches else 0.0

            if save_strategy == "best_val" and mean_iou > val_iou_best:
                val_iou_best = mean_iou
                # save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }, save_path)
            msg += f" - val mIoU: {mean_iou:.4f}"
        print(msg)


    return model


if __name__ == '__main__':
    args_parser = ArgumentParser()
    args_parser.add_argument('--data_dir', type=str, default='data/segmentation', help='Path to prepared data directory')
    args_parser.add_argument('--zonal_labels_file', type=str, default='data/segmentation/zonal_labels.txt', help='Path to zonal labels text file')
    args_parser.add_argument('--oversampling_ratio', type=float, default=1.0, help='Oversampling ratio for class balancing')
    args_parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    args_parser.add_argument('--num_epochs', type=int, default=4, help='Number of training epochs')
    args_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args_parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    args_parser.add_argument('--image_size', type=tuple, default=(512, 512), help='Input image size')
    args_parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    args_parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to model checkpoint to resume training')
    args = args_parser.parse_args()


    train_img_dir = os.path.join(args.data_dir, 'train', 'images')
    train_mask_dir = os.path.join(args.data_dir, 'train', 'masks')
    val_img_dir = os.path.join(args.data_dir, 'val', 'images')
    val_mask_dir = os.path.join(args.data_dir, 'val', 'masks')
    # load zonal labels
    with open(args.zonal_labels_file, 'r') as f:
        zonal_labels = [line.strip() for line in f.readlines()]

    model = train_segmentation(
        train_images_dir=train_img_dir,
        train_masks_dir=train_mask_dir,
        val_images_dir=val_img_dir,
        val_masks_dir=val_mask_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        pretrained=args.pretrained,
        num_epochs=args.num_epochs,
        save_path="deeplab_custom.pth",
        zonal_labels=zonal_labels,
        oversampling_ratio=args.oversampling_ratio,
        checkpoint_path=args.checkpoint_path,
    )