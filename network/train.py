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
import gc
import argparse

from netmodel import NeuroImagingNet

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
    def __init__(self, temporal_model="pooled"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.grad_kernel = torch.tensor(
            [[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]]
        ).float()

    def gradient_loss(self, pred, target):
        pred_grad, target_grad = None, None
        if temporal_model == "pooled":
            pred_grad = F.conv2d(
                pred.unsqueeze(1), self.grad_kernel.to(pred.device)
            )
            target_grad = F.conv2d(
                target.unsqueeze(1), self.grad_kernel.to(target.device)
            )
        else:
            # seq模式 [B,T,H,W]
            pred_grad = F.conv2d(pred, self.grad_kernel.to(pred.device))
            target_grad = F.conv2d(target, self.grad_kernel.to(target.device))
        return F.l1_loss(pred_grad, target_grad)

    def projection_loss(self, pred, target):
        """计算x轴投影的MSE损失（自动归一化）"""
        if temporal_model == "pooled":
            # pooled模式 [B,T,H,W]
            pred_2d = pred.unsqueeze(1)
            target_2d = target.unsqueeze(1)
        else:
            # seq模式 [B,T,H,W]
            pred_2d = pred
            target_2d = target

        # 计算x轴投影（沿高度方向求和）
        pred_proj = pred_2d.sum(dim=2)  # 形状 [B, 1, W]
        target_proj = target_2d.sum(dim=2)

        # 计算归一化投影损失
        proj_loss = F.mse_loss(pred_proj, target_proj)
        return proj_loss / 1000

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        if temporal_model == "pooled":
            # pooled模式 [B,T,H,W]
            ssim_loss = 1 - self.ssim(pred.unsqueeze(1), target.unsqueeze(1))
        else:
            # seq模式 [B,T,H,W]
            ssim_loss = 1 - self.ssim(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        proj_loss = self.projection_loss(pred, target)
        return (
            0.6 * mse_loss + 0.3 * ssim_loss + 0.1 * grad_loss + 0.1 * proj_loss
        )


# 完整训练流程
def train_reconstruction(
    projections,
    images,
    enhanced=False,
    temporal_model="pooled",
    batch_size=64,
    val_ratio=0.2,
):
    # 划分训练验证集
    dataset_size = len(projections)
    print(f"数据集大小: {dataset_size}")
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
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    # 初始化模型和优化器
    model = NeuroImagingNet(enhanced=enhanced, temporal_mode=temporal_model).to(
        device
    )
    # 损失函数 由于函数设计问题及满足时间聚合情况下的兼容,构建不同函数
    criterion = ReconLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=7,
        threshold=1e-5,
        min_lr=1e-7,
        verbose=True,
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

    # 使用tqdm创建主进度条

    with tqdm(range(epochs), desc="Training Progress", unit="epoch") as pbar:
        for epoch in pbar:
            # 训练阶段
            model.train()
            train_loss = 0.0

            # 添加训练批次进度条 - 注意leave=False确保它不会保留
            for i, (proj, img) in enumerate(
                tqdm(
                    train_loader,
                    desc=f"Train - Epoch {epoch + 1}/{epochs}",
                    leave=False,
                    unit="batch",
                )
            ):
                proj, img = proj.to(device), img.to(device)

                optimizer.zero_grad()
                output = model(proj)  # [B, T, H, W]
                loss = criterion(output, img)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * proj.size(0)

            # 验证阶段
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for proj, img in tqdm(
                    val_loader,
                    desc=f"Valid - Epoch {epoch + 1}/{epochs}",
                    leave=False,
                    unit="batch",
                ):
                    proj, img = proj.to(device), img.to(device)
                    output = model(proj)
                    batch_loss = criterion(output, img).item()
                    val_loss += batch_loss * proj.size(0)

            # 计算平均损失
            train_loss /= len(train_set)
            val_loss /= len(val_set)

            # 学习率调度
            scheduler.step(val_loss)

            # 记录损失和学习率
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            learning_rates.append(optimizer.param_groups[0]["lr"])

            # 更新主进度条
            pbar.set_postfix(
                {
                    "t_loss": f"{train_loss:.4f}",
                    "v_loss": f"{val_loss:.4f}",
                    "best": f"{best_loss:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            # 早停与保存
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), "model_peremeter.pth")
                no_improve = 0
                pbar.set_description(f"Training Progress (Best: {epoch + 1})")
            else:
                # 若处于调节冷却期则不进行处理 暂定冷却期为5
                if (
                    len(learning_rates) > 5
                    and learning_rates[-1] != learning_rates[-2]
                ):
                    no_improve -= 5
                else:
                    no_improve += 1
                if no_improve >= patience:
                    pbar.write(f"Early stopping at epoch {epoch + 1}")
                    break

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


if __name__ == "__main__":

    # 改为命令行方式
    parser = argparse.ArgumentParser(description="重建模型训练和测试")
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="测试模式，仅加载data_img0.mat数据，默认非测试模式",
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        default=False,
        help="增强模式 完整注意力模式 (默认非增强模式)",
    )
    parser.add_argument(
        "--temporal_model",
        type=str,
        default="pooled",
        choices=["pooled", "seq"],
        help="时间模型类型 (pooled/seq)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument(
        "--chunk_size", type=int, default=64, help="数据分块大小"
    )
    args = parser.parse_args()

    # 加载设置
    torch.backends.cudnn.benchmark = True  # 加速卷积运算
    torch.cuda.empty_cache()  # 清空缓存
    # 防显存碎片
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 使用命令行参数替代硬编码值
    test_mode = args.test
    enhanced = args.enhanced
    temporal_model = args.temporal_model
    batch_size = args.batch_size
    chunk_size = args.chunk_size

    # 数据导入
    data_dir = "./data"  # 数据文件所在目录
    projections = []
    images = []
    slip_data = []
    if args.test:
        slip_data = ["data_real3.mat"]
    else:
        slip_data = ["data_img0.mat"]

    if not os.path.exists(data_dir):
        print(f"目录 {data_dir} 不存在")
    else:
        mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
        if not mat_files:
            print(f"在 {data_dir} 中未找到.mat文件")
        else:
            mat_files.sort()  # 顺序处理
            for filename in mat_files:
                if filename in slip_data:
                    print(f"配置跳过文件: {filename}")
                    continue
                file_path = os.path.join(data_dir, filename)
                try:
                    with h5py.File(file_path, "r") as file:
                        # 数据格式
                        print(f"正在读取文件: {filename}")
                        print(f"文件内容: {list(file.keys())}")

                        # 检查图像数据变量名
                        img_var = None
                        if "all_imgs" in file:
                            img_var = "all_imgs"
                        elif "all_ans" in file:
                            img_var = "all_ans"
                        else:
                            print(
                                f"文件 {filename} 中未找到 all_imgs 或 all_ans 键"
                            )
                            continue

                        # 获取数据形状
                        proj_shape = file["all_proj"].shape
                        img_shape = file[img_var].shape
                        num_samples = proj_shape[2]  # 第三维是样本数
                        print(f"投影数据形状: {proj_shape}")
                        print(f"图像数据形状: {img_shape}")

                        # 分段读取 图像数据大小为[B,T,H,W]很大 mat格式中倒转
                        # 读取并转置 images
                        # 分块读取数据
                        num_chunks = (
                            num_samples + chunk_size - 1
                        ) // chunk_size
                        print(f"分 {num_chunks} 块读取文件 {filename}")

                        # 单文件数据
                        file_proj = []
                        file_imgs = []

                        for i in range(num_chunks):
                            # 测试 小批量加载
                            if i >= 5:
                                break
                            start_idx = i * chunk_size
                            end_idx = min(start_idx + chunk_size, num_samples)
                            samples_in_chunk = end_idx - start_idx

                            print(
                                f"读取块 {i+1}/{num_chunks} (样本 {start_idx}-{end_idx})"
                            )

                            # 读取投影数据块
                            a_proj = file["all_proj"][:, :, start_idx:end_idx]
                            a_proj = np.transpose(a_proj, (2, 1, 0))
                            file_proj.append(a_proj)

                            # 读取图像数据块 针对pooled模式和seq模式分开
                            if test_mode:
                                a_imgs = file[img_var][:, :, start_idx:end_idx]
                                a_imgs = np.transpose(a_imgs, (2, 1, 0))
                            else:
                                # 全集数据格式
                                a_imgs = file[img_var][
                                    :, :, :, start_idx:end_idx
                                ]
                                a_imgs = np.transpose(a_imgs, (3, 2, 1, 0))
                            file_imgs.append(a_imgs)

                        # 合并文件的所有块
                        file_proj = np.concatenate(file_proj, axis=0)
                        file_imgs = np.concatenate(file_imgs, axis=0)

                        # 合并到总数据
                        projections.append(file_proj)
                        images.append(file_imgs)

                        # 清理临时变量
                        del file_proj, file_imgs
                        gc.collect()

                    print(
                        f"已加载 {len(projections)} 个样本，来自文件 {filename}"
                    )

                except Exception as e:
                    print(f"读取文件 {filename} 时出错: {str(e)}")
                    continue
    # 合并所有数据
    if projections:
        # 沿第一个维度（axis=0）合并
        projections = np.concatenate(projections, axis=0)
        images = np.concatenate(images, axis=0)

        print("projections shape:", projections.shape)
        print("images shape:", images.shape)
        print(f"total samples: {len(projections)}")
    else:
        print("no files found")

    # 训练
    print("Starting training...")
    trained_model = train_reconstruction(
        projections,
        images,
        enhanced=enhanced,
        temporal_model=temporal_model,
        batch_size=batch_size,
    )

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
