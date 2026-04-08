# Copyright (c) 2026 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 导入我们动过外科手术的 MODNet 和训练引擎
from src.models.modnet import MODNet
from src.trainer import supervised_training_iter

# ==========================================
# 1. 动态 Trimap 生成与数据管道 (Block 1.0)
# ==========================================
class P3MDataset(Dataset):
    def __init__(self, image_dir, mask_dir, input_size=512):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = input_size

        # 获取所有 jpg 文件名
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        print(f"[Dataset] 成功加载 {len(self.image_names)} 张训练样本。")

        # 定义图像的归一化 (MODNet 通常需要 -1 到 1 的分布)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_names)

    def _generate_trimap(self, alpha):
        """ 核心魔法：用数学形态学动态生成 Trimap """
        # 随机内核大小，让模型适应不同宽度的未知边缘
        k_size = random.choice(range(10, 30))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        dilated = cv2.dilate(alpha, kernel, iterations=1)
        eroded = cv2.erode(alpha, kernel, iterations=1)

        # 0=背景, 1=前景, 0.5=未知边缘区
        trimap = np.full(alpha.shape, 0.5, dtype=np.float32)
        trimap[eroded >= 255] = 1.0
        trimap[dilated <= 0] = 0.0
        return trimap

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        mask_name = img_name.replace('.jpg', '.png')

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 1. 读取并缩放 RGB 图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        # 2. 读取、缩放并单通道化 Mask
        # 你之前的测试显示它是 3 通道，我们用 IMREAD_GRAYSCALE 强转为 1 通道
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        # 3. 动态生成 Trimap
        trimap = self._generate_trimap(mask)

        # 4. 格式化给 PyTorch
        # Image 经过 transform 已经变成 [3, H, W] tensor
        image_tensor = self.img_transform(image)

        # Mask 转为 [1, H, W] 且范围在 [0.0, 1.0]
        gt_matte_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        # Trimap 转为 [1, H, W]
        trimap_tensor = torch.from_numpy(trimap).unsqueeze(0).float()

        return image_tensor, trimap_tensor, gt_matte_tensor

# ==========================================
# 2. 训练主循环引擎 (Block 1.1 验证)
# ==========================================
def main():
    # 强制路径：请根据你的服务器实际情况调整
    TRAIN_IMAGE_DIR = "/development/docker_volumes/src/ai/image-matting/primary-folder/dataset/P3M-10k/train/blurred_image"
    TRAIN_MASK_DIR = "/development/docker_volumes/src/ai/image-matting/primary-folder/dataset/P3M-10k/train/mask"

    BATCH_SIZE = 4  # MVP 阶段先设小一点，防显存炸裂
    EPOCHS = 1      # 我们先只跑 1 个 epoch 验证能跑通
    LR = 0.01

    print("🚀 [Phase 1.0] 初始化数据管道...")
    dataset = P3MDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, input_size=512)
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    print("🚀 [Phase 1.1] 实例化纯 BN 版 MODNet...")
    # backbone_pretrained=False，因为我们彻底改了结构，需要用数据从头磨合
    modnet = MODNet(backbone_pretrained=False)
    # modnet = nn.DataParallel(modnet).cuda() # 如果你有多个 GPU，取消注释这行，注释掉下一行
    modnet = modnet.cuda()

    optimizer = torch.optim.SGD(modnet.parameters(), lr=LR, momentum=0.9)

    print("🔥 开始强袭训练 (MVP 验证测试)...")
    for epoch in range(EPOCHS):
        for idx, (image, trimap, gt_matte) in enumerate(dataloader):
            # 将数据推入 GPU
            image = image.cuda()
            trimap = trimap.cuda()
            gt_matte = gt_matte.cuda()

            # 调用引擎的单步迭代
            semantic_loss, detail_loss, matte_loss = supervised_training_iter(
                modnet, optimizer, image, trimap, gt_matte
            )

            total_loss = semantic_loss + detail_loss + matte_loss

            if idx % 10 == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}][Batch {idx}/{len(dataloader)}] "
                      f"Total Loss: {total_loss.item():.4f} "
                      f"(Sem: {semantic_loss.item():.4f}, Det: {detail_loss.item():.4f}, Mat: {matte_loss.item():.4f})")

            # 作为 MVP 测试，跑 50 个 batch 不报错我们就主动打断退出，宣布 Block 1 成功
            if idx == 50:
                print("🎯 MVP 验证通过！网络结构和数据管道运转完美！即将保存测试 Checkpoint。")
                break

    # 存下一个模型，为下一个 Block (ONNX 导出) 储备子弹
    torch.save(modnet.state_dict(), 'modnet_pure_bn_mvp.ckpt')
    print("💾 模型已保存至 modnet_pure_bn_mvp.ckpt")

if __name__ == '__main__':
    main()