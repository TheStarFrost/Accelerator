# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
import h5py
import os
from tqdm import tqdm

from netmodel_v3 import NeuroImagingNet

matplotlib.use("Agg")  # 设置为无头模式

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 物理参数配置
PHYSICS = {
    "num_angles": 128,  # 投影角度数
    "detectors": 128,  # 探测器数量
    "image_size": (128, 128),  # 重建图像尺寸
}


# 数据预处理类
class DataProcessor:
    """投影数据处理器"""

    def __init__(self, physics_params):
        self.physics = physics_params

    def __call__(self, raw_projection, gt_image):
        """
        输入参数：
        raw_projection: 原始投影数据 [num_angles, detectors]
        gt_image: 对应的真实图像 [H, W]
        """

        # 投影数据归一化
        projection = self._normalize_projection(raw_projection)

        # 图像数据归一化
        image = self._normalize_image(gt_image)

        return projection, image

    def _normalize_projection(self, projection):
        """投影数据标准化"""
        # return (projection - np.mean(projection)) / (np.std(projection) + 1e-8)
        return projection

    def _normalize_image(self, image):
        """图像数据归一化到[-1, 1]"""
        return (image / image.max()) * 2 - 1


# 自定义数据集类
class ReconDataset(data.Dataset):
    def __init__(self, projections, images, transform=None):
        """
        参数：
        projections: 投影数据 [N, num_angles, detectors]
        images: 对应重建图像 [N, H, W]
        """
        self.projections = projections
        self.images = images
        self.processor = DataProcessor(PHYSICS)

    def __len__(self):
        return len(self.projections)

    def __getitem__(self, idx):
        proj = self.projections[idx]
        img = self.images[idx]

        # 物理预处理
        proj, img = self.processor(proj, img)

        return torch.FloatTensor(proj), torch.FloatTensor(img)


# 复合损失函数
class ReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.grad_kernel = torch.tensor(
            [[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]]
        ).float()

    def gradient_loss(self, pred, target):
        pred_grad = F.conv2d(
            pred.unsqueeze(1), self.grad_kernel.to(pred.device)
        )
        target_grad = F.conv2d(
            target.unsqueeze(1), self.grad_kernel.to(target.device)
        )
        return F.l1_loss(pred_grad, target_grad)

    def projection_loss(self, pred, target):
        """计算x轴投影的MSE损失（自动归一化）"""
        pred_2d = pred.unsqueeze(1)  # [B, 1, H, W]
        target_2d = target.unsqueeze(1)

        # 计算x轴投影（沿高度方向求和）
        pred_proj = pred_2d.sum(dim=2)  # 形状 [B, 1, W]
        target_proj = target_2d.sum(dim=2)

        # 计算归一化投影损失
        proj_loss = F.mse_loss(pred_proj, target_proj)
        return proj_loss / 1000

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_loss = 1 - self.ssim(pred.unsqueeze(1), target.unsqueeze(1))
        grad_loss = self.gradient_loss(pred, target)
        proj_loss = self.projection_loss(pred, target)
        return (
            0.6 * mse_loss + 0.3 * ssim_loss + 0.1 * grad_loss + 0.1 * proj_loss
        )


# 完整训练流程
def train_reconstruction(projections, images, val_ratio=0.2):
    # 划分训练验证集
    dataset_size = len(projections)
    # np.random.seed(88)
    indices = np.random.permutation(dataset_size)
    split = int(np.floor(val_ratio * dataset_size))

    train_set = ReconDataset(
        projections[indices[split:]], images[indices[split:]]
    )
    val_set = ReconDataset(
        projections[indices[:split]], images[indices[:split]]
    )

    # 创建数据加载器
    train_loader = data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=16, pin_memory=True
    )
    val_loader = data.DataLoader(
        val_set, batch_size=128, shuffle=False, num_workers=16, pin_memory=True
    )

    # 初始化模型和优化器
    model = NeuroImagingNet().to(device)
    criterion = ReconLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 添加记录
    train_losses = []  # 存储每个epoch的训练损失
    val_losses = []  # 存储每个epoch的验证损失
    learning_rates = []  # 存储每个epoch的学习率

    # 训练循环
    best_loss = float("inf")
    patience = 20
    no_improve = 0
    epochs = 256  # 设置总epochs数

    # 使用tqdm创建进度条
    with tqdm(
        range(epochs),
        desc="Training Progress",
        unit="epoch",
        dynamic_ncols=False,
    ) as pbar:
        for epoch in pbar:

            # 训练阶段
            model.train()
            train_loss = 0.0

            # 添加训练批次进度条
            train_batches = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False,
                unit="batch",
            )

            for i, (proj, img) in enumerate(train_loader):
                proj, img = proj.to(device), img.to(device)

                optimizer.zero_grad()
                output = model(proj)
                loss = criterion(output, img)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * proj.size(0)

                # 更新批次进度条
                train_batches.set_postfix(loss=loss.item())

            # 验证阶段
            model.eval()
            val_loss = 0.0

            # 添加验证批次进度条
            val_batches = tqdm(
                val_loader, desc="Validating", leave=False, unit="batch"
            )

            with torch.no_grad():
                for proj, img in val_loader:
                    proj, img = proj.to(device), img.to(device)
                    output = model(proj)
                    val_loss += criterion(output, img).item() * proj.size(0)

                    # 更新验证批次进度条
                    val_batches.set_postfix(loss=loss.item())

            # 关闭批次进度条
            train_batches.close()
            val_batches.close()

            # 计算平均损失
            train_loss /= len(train_set)
            val_loss /= len(val_set)

            # 学习率调度
            scheduler.step(val_loss)

            # 记录损失和学习率
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            learning_rates.append(optimizer.param_groups[0]["lr"])  # 当前学习率

            # 更新主进度条
            pbar.set_postfix(
                {
                    "Train Loss": f"{train_loss:.4f}",
                    "Val Loss": f"{val_loss:.4f}",
                    "Best Val Loss": f"{best_loss:.4f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            # 早停与保存
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), "model_peremeter.pth")
                no_improve = 0
                pbar.set_description(
                    f"Training Progress (Best Epoch: {epoch + 1})"
                )
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            print(
                f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 学习率曲线
    plt.subplot(1, 2, 2)
    plt.plot(learning_rates, label="Learning Rate", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()

    print("Training Complete!")
    return model


# 测试示例（假设已有数据）
if __name__ == "__main__":
    # 导入训练数据
    all_projections = []
    all_images = []

    # 假设文件命名是连续的：data0.mat, data1.mat, data2.mat...
    data_dir = "./data"  # 数据文件所在目录
    i = 0

    while True:
        filename = os.path.join(data_dir, f"data_img{i}.mat")

        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"file {filename} donot exist，stop reading")
            break

        # 读取当前文件
        with h5py.File(filename, "r") as file:
            # 读取并转置 projections
            a_projections = file["all_proj"][()]
            a_projections = np.transpose(a_projections, (2, 1, 0))
            all_projections.append(a_projections)

            # 读取并转置 images
            a_images = file["all_imgs"][()]
            a_images = np.transpose(a_images, (2, 1, 0))
            all_images.append(a_images)

        print(f"loaded {len(a_projections)} samples in {filename} ")
        i += 1  # 递增索引，读取下一个文件

    # #自动读取data文件夹下所有.mat文件
    # data_dir = "./data"  # 数据文件所在目录
    # all_projections = []
    # all_images = []

    # # 确保目录存在
    # if not os.path.exists(data_dir):
    #     print(f"目录 {data_dir} 不存在")
    # else:
    #     # 获取所有.mat文件
    #     mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

    #     if not mat_files:
    #         print(f"在 {data_dir} 中未找到.mat文件")
    #     else:
    #         # 排序文件名，确保按顺序处理
    #         mat_files.sort()

    #         # 遍历每个文件
    #         for filename in mat_files:
    #             file_path = os.path.join(data_dir, filename)

    #             try:
    #                 # 读取当前文件
    #                 with h5py.File(file_path, "r") as file:
    #                     # 读取并转置 projections
    #                     a_projections = file["all_proj"][()]
    #                     a_projections = np.transpose(a_projections, (2, 1, 0))
    #                     all_projections.append(a_projections)

    #                     # 读取并转置 images
    #                     a_images = file["all_imgs"][()]
    #                     a_images = np.transpose(a_images, (2, 1, 0))
    #                     all_images.append(a_images)

    #                 print(f"已加载 {len(a_projections)} 个样本，来自文件 {filename}")

    #             except Exception as e:
    #                 print(f"读取文件 {filename} 时出错: {str(e)}")
    #                 continue

    # 合并所有数据
    if all_projections:
        # 沿第一个维度（axis=0）合并
        projections = np.concatenate(all_projections, axis=0)
        images = np.concatenate(all_images, axis=0)

        print("projections shape:", projections.shape)
        print("images shape:", images.shape)
        print(f"total samples: {len(projections)}")
    else:
        print("no files found")

    # 运行训练
    print("Starting training...")
    # 显存优化配置
    torch.backends.cudnn.benchmark = True  # 加速卷积运算
    torch.cuda.empty_cache()  # 清空缓存
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True"  # 防显存碎片
    )
    trained_model = train_reconstruction(projections, images)

    # 无头模式存储可视化
    print("\nGenerating output images...")
    vmin, vmax = -1, 1
    output_folder = "output_images"  # 输出文件夹路径

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 创建画布和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for test_idx in range(64):
        # 测试推理
        test_proj = (
            torch.FloatTensor(projections[test_idx]).unsqueeze(0).to(device)
        )
        reconstructed = (
            trained_model(test_proj).squeeze().cpu().detach().numpy()
        )
        ground_truth = (images[test_idx] / images[test_idx].max()) * 2 - 1

        # 更新图像数据
        ax1.imshow(ground_truth, cmap="gray", vmin=vmin, vmax=vmax)
        ax1.set_title(f"Ground Truth ({test_idx + 1})")
        ax1.axis("off")

        ax2.imshow(reconstructed, cmap="gray", vmin=vmin, vmax=vmax)
        ax2.set_title(f"Reconstructed (t={test_idx + 1})")
        ax2.axis("off")

        # 保存图像到文件夹
        plt.savefig(os.path.join(output_folder, f"frame_{test_idx:03d}.png"))

        # 清除当前子图内容，为下一帧做准备
        ax1.clear()
        ax2.clear()

    plt.close(fig)  # 关闭画布

    print("\nAll operations completed!")
