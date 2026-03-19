"""
Block 1.2: Fine-tune MODNet (Pure BatchNorm) on P3M-10k
=========================================================
Strategy: Fast Validation (15 epochs, pretrained backbone, 512x512)

Key design decisions:
  - backbone_pretrained=True: Load ImageNet-trained MobileNetV2 weights
    (the backbone has NO IBNorm, so original weights are fully compatible)
  - 512x512 input: Train at high-res so the model learns hair edge features,
    even though NPU deployment will use 256x256 (Phase 3 Guided Filter recovers detail)
  - StepLR(step=5, gamma=0.1): Start with lr=0.01, decay at epoch 5/10/15
  - Per-epoch validation: Save visual comparison (Input | Prediction | GT) to output/
  - Save best checkpoint by validation loss

Usage:
  cd helmsman.git/third-party/sdk/MODNet.git
  python3 -m train_modnet_block1_2

Author: Claude (AI Copilot) + Potter White
Date: 2026-03-19
Phase: 1 -> Block 1.2 (Fine-tune Training)
"""

import os
import cv2
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.models.modnet import MODNet
from src.trainer import supervised_training_iter


# ==============================================================================
# 1. Dataset with Dynamic Trimap Generation & Data Augmentation
# ==============================================================================

class P3MDataset(Dataset):
    """P3M-10k dataset loader with dynamic trimap generation.

    Each sample returns: (image_tensor, trimap_tensor, gt_matte_tensor)
      - image_tensor: [3, H, W], normalized to [-1, 1]
      - trimap_tensor: [1, H, W], values in {0.0, 0.5, 1.0}
      - gt_matte_tensor: [1, H, W], values in [0.0, 1.0]

    The trimap is generated on-the-fly using morphological dilation/erosion
    on the ground truth mask. This forces the model to focus on learning
    the uncertain boundary region (hair strands, semi-transparent edges).
    """

    def __init__(self, image_dir, mask_dir, input_size=512, augment=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.augment = augment

        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        print(f"[Dataset] Loaded {len(self.image_names)} samples from {image_dir}")

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Color jitter for augmentation (only applied to RGB, not mask)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )

    def __len__(self):
        return len(self.image_names)

    def _generate_trimap(self, alpha):
        """Generate trimap from alpha mask using morphological operations.

        Args:
            alpha: uint8 grayscale mask, values in [0, 255]

        Returns:
            trimap: float32 array, values in {0.0, 0.5, 1.0}
                    0.0 = definite background
                    0.5 = unknown/boundary region (where the AI must focus)
                    1.0 = definite foreground
        """
        k_size = random.choice(range(10, 30))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        dilated = cv2.dilate(alpha, kernel, iterations=1)
        eroded = cv2.erode(alpha, kernel, iterations=1)

        trimap = np.full(alpha.shape, 0.5, dtype=np.float32)
        trimap[eroded >= 255] = 1.0
        trimap[dilated <= 0] = 0.0
        return trimap

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        mask_name = img_name.replace('.jpg', '.png')

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Read and resize
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_LINEAR)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
        mask = cv2.resize(mask, (self.input_size, self.input_size),
                          interpolation=cv2.INTER_LINEAR)

        # Data augmentation: random horizontal flip (applied to BOTH image and mask)
        if self.augment and random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Generate trimap from mask
        trimap = self._generate_trimap(mask)

        # Convert image to tensor with normalization
        # If augmenting, apply color jitter BEFORE ToTensor
        from PIL import Image
        pil_image = Image.fromarray(image)
        if self.augment:
            pil_image = self.color_jitter(pil_image)
        image_tensor = self.img_transform(pil_image)

        # Mask -> [1, H, W], range [0.0, 1.0]
        gt_matte_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        # Trimap -> [1, H, W]
        trimap_tensor = torch.from_numpy(trimap).unsqueeze(0).float()

        return image_tensor, trimap_tensor, gt_matte_tensor


class P3MValDataset(Dataset):
    """Validation dataset (no augmentation, returns filename for visualization)."""

    def __init__(self, image_dir, mask_dir, input_size=512):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = input_size

        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        print(f"[ValDataset] Loaded {len(self.image_names)} validation samples")

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        mask_name = img_name.replace('.jpg', '.png')

        image = cv2.imread(os.path.join(self.image_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_LINEAR)

        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.input_size, self.input_size),
                          interpolation=cv2.INTER_LINEAR)

        image_tensor = self.img_transform(image)
        gt_matte_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        # Also return the raw image (uint8 RGB) for visualization
        return image_tensor, gt_matte_tensor, image, img_name


# ==============================================================================
# 2. Validation & Visualization
# ==============================================================================

def validate(modnet, val_loader, device, epoch, output_dir, num_vis=3):
    """Run validation and save visual comparison images.

    Returns:
        avg_loss: Average L1 loss on validation set (lower = better)
    """
    modnet.eval()
    total_loss = 0.0
    count = 0
    vis_images = []

    with torch.no_grad():
        for idx, (image_tensor, gt_matte, raw_image, img_name) in enumerate(val_loader):
            image_tensor = image_tensor.to(device)
            gt_matte = gt_matte.to(device)

            # Forward pass in inference mode
            _, _, pred_matte = modnet(image_tensor, True)

            # Compute L1 loss
            loss = F.l1_loss(pred_matte, gt_matte)
            total_loss += loss.item() * image_tensor.size(0)
            count += image_tensor.size(0)

            # Collect first N samples for visualization
            if len(vis_images) < num_vis:
                for b in range(min(image_tensor.size(0), num_vis - len(vis_images))):
                    vis_images.append({
                        'raw': raw_image[b].numpy() if isinstance(raw_image[b], torch.Tensor) else raw_image[b],
                        'pred': pred_matte[b, 0].cpu().numpy(),
                        'gt': gt_matte[b, 0].cpu().numpy(),
                        'name': img_name[b],
                    })

    avg_loss = total_loss / max(count, 1)

    # Save visualization
    if vis_images:
        _save_visualization(vis_images, epoch, output_dir, avg_loss)

    modnet.train()
    return avg_loss


def _save_visualization(vis_images, epoch, output_dir, avg_loss):
    """Save a comparison image: Input | Prediction | Ground Truth (per row)."""
    os.makedirs(output_dir, exist_ok=True)

    rows = len(vis_images)
    cell_h, cell_w = 256, 256  # visualization cell size
    canvas = np.zeros((rows * cell_h + 40, cell_w * 3 + 20, 3), dtype=np.uint8)

    # Header
    cv2.putText(canvas, f"Epoch {epoch+1} | Val L1 Loss: {avg_loss:.4f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for i, vis in enumerate(vis_images):
        y_offset = 40 + i * cell_h

        # Input image (RGB)
        raw = vis['raw']
        if isinstance(raw, np.ndarray):
            raw_resized = cv2.resize(raw, (cell_w, cell_h))
            raw_bgr = cv2.cvtColor(raw_resized, cv2.COLOR_RGB2BGR)
        else:
            raw_bgr = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        canvas[y_offset:y_offset+cell_h, 0:cell_w] = raw_bgr

        # Prediction (grayscale -> colormap for visibility)
        pred = (vis['pred'] * 255).clip(0, 255).astype(np.uint8)
        pred_resized = cv2.resize(pred, (cell_w, cell_h))
        pred_color = cv2.cvtColor(pred_resized, cv2.COLOR_GRAY2BGR)
        canvas[y_offset:y_offset+cell_h, cell_w+10:cell_w*2+10] = pred_color

        # Ground truth
        gt = (vis['gt'] * 255).clip(0, 255).astype(np.uint8)
        gt_resized = cv2.resize(gt, (cell_w, cell_h))
        gt_color = cv2.cvtColor(gt_resized, cv2.COLOR_GRAY2BGR)
        canvas[y_offset:y_offset+cell_h, cell_w*2+20:cell_w*3+20] = gt_color

    # Labels at bottom
    label_y = 40 + rows * cell_h - 10
    # (labels would be cut off, put them at top of each column instead)

    out_path = os.path.join(output_dir, f"epoch_{epoch+1:02d}_val.png")
    cv2.imwrite(out_path, canvas)
    print(f"  [Viz] Saved validation preview -> {out_path}")


# ==============================================================================
# 3. Training Main Loop
# ==============================================================================

def main():
    # ===== Configuration =====
    TRAIN_IMAGE_DIR = "/development/docker_volumes/src/ai/image-matting/primary-folder/dataset/P3M-10k/train/blurred_image"
    TRAIN_MASK_DIR  = "/development/docker_volumes/src/ai/image-matting/primary-folder/dataset/P3M-10k/train/mask"
    VAL_IMAGE_DIR   = "/development/docker_volumes/src/ai/image-matting/primary-folder/dataset/P3M-10k/validation/P3M-500-P/blurred_image"
    VAL_MASK_DIR    = "/development/docker_volumes/src/ai/image-matting/primary-folder/dataset/P3M-10k/validation/P3M-500-P/mask"

    OUTPUT_DIR     = "./output"
    CKPT_DIR       = "./checkpoints"

    BATCH_SIZE     = 8       # BN needs >= 4; 8 is stable on 24GB GPU
    EPOCHS         = 15      # Fast validation strategy
    LR             = 0.01    # Will decay via StepLR
    INPUT_SIZE     = 512     # Train at 512 for edge quality
    NUM_WORKERS    = 4       # DataLoader workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] Device: {device}")
    print(f"[Config] Epochs: {EPOCHS}, BS: {BATCH_SIZE}, LR: {LR}, Input: {INPUT_SIZE}x{INPUT_SIZE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ===== Data =====
    print("\n[Phase 1.2] Initializing data pipeline...")
    train_dataset = P3MDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR,
                               input_size=INPUT_SIZE, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              drop_last=True, pin_memory=True)

    val_dataset = P3MValDataset(VAL_IMAGE_DIR, VAL_MASK_DIR,
                                input_size=INPUT_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=2)

    # ===== Model =====
    print("\n[Phase 1.2] Instantiating Pure-BN MODNet with pretrained backbone...")
    modnet = MODNet(backbone_pretrained=True)
    modnet = modnet.to(device)

    # Print parameter count
    total_params = sum(p.numel() for p in modnet.parameters())
    trainable_params = sum(p.numel() for p in modnet.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total_params:,}")
    print(f"[Model] Trainable parameters: {trainable_params:,}")

    # ===== Optimizer & Scheduler =====
    optimizer = torch.optim.SGD(modnet.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # ===== Training Loop =====
    best_val_loss = float('inf')

    print(f"\n{'='*60}")
    print(f"  BLOCK 1.2: Fine-tune Training Started")
    print(f"  {len(train_dataset)} train samples, {len(val_dataset)} val samples")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        modnet.train()
        epoch_sem_loss = 0.0
        epoch_det_loss = 0.0
        epoch_mat_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch+1}/{EPOCHS}] LR: {current_lr:.6f}")

        for idx, (image, trimap, gt_matte) in enumerate(train_loader):
            image = image.to(device)
            trimap = trimap.to(device)
            gt_matte = gt_matte.to(device)

            semantic_loss, detail_loss, matte_loss = supervised_training_iter(
                modnet, optimizer, image, trimap, gt_matte
            )

            total_loss = semantic_loss + detail_loss + matte_loss
            epoch_sem_loss += semantic_loss.item()
            epoch_det_loss += detail_loss.item()
            epoch_mat_loss += matte_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1

            if idx % 50 == 0:
                print(f"  [Batch {idx:4d}/{len(train_loader)}] "
                      f"Loss: {total_loss.item():.4f} "
                      f"(Sem: {semantic_loss.item():.4f}, "
                      f"Det: {detail_loss.item():.4f}, "
                      f"Mat: {matte_loss.item():.4f})")

        # Epoch summary
        avg_total = epoch_total_loss / num_batches
        avg_sem = epoch_sem_loss / num_batches
        avg_det = epoch_det_loss / num_batches
        avg_mat = epoch_mat_loss / num_batches
        print(f"  [Epoch {epoch+1} Summary] Avg Loss: {avg_total:.4f} "
              f"(Sem: {avg_sem:.4f}, Det: {avg_det:.4f}, Mat: {avg_mat:.4f})")

        # Step the learning rate scheduler
        scheduler.step()

        # Validation
        print(f"  [Validation] Running on {len(val_dataset)} samples...")
        val_loss = validate(modnet, val_loader, device, epoch, OUTPUT_DIR, num_vis=3)
        print(f"  [Validation] Avg L1 Loss: {val_loss:.4f}")

        # Save checkpoint every epoch
        ckpt_path = os.path.join(CKPT_DIR, f"modnet_bn_epoch_{epoch+1:02d}.ckpt")
        torch.save(modnet.state_dict(), ckpt_path)
        print(f"  [Checkpoint] Saved -> {ckpt_path}")

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CKPT_DIR, "modnet_bn_best.ckpt")
            torch.save(modnet.state_dict(), best_path)
            print(f"  [Best Model] New best! Val Loss: {val_loss:.4f} -> {best_path}")

        print()

    # ===== Final Summary =====
    print(f"{'='*60}")
    print(f"  BLOCK 1.2 TRAINING COMPLETE")
    print(f"  Best Validation L1 Loss: {best_val_loss:.4f}")
    print(f"  Best checkpoint: {CKPT_DIR}/modnet_bn_best.ckpt")
    print(f"  Visual results: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
