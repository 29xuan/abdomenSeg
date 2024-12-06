import os
from monai.transforms import LoadImaged, EnsureChannelFirstd, Compose
from monai.data import DataLoader, Dataset
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd


# 绘制损失曲线的函数
def plot_losses(output_path, epoch_loss_values, val_loss_values, epoch_cl_loss_values, epoch_recon_loss_values, epoch):
    plt.figure(figsize=(10, 8))

    # 绘制训练损失
    plt.subplot(2, 2, 1)
    plt.plot(epoch_loss_values, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.legend()

    # 绘制验证损失
    plt.subplot(2, 2, 2)
    plt.plot(val_loss_values, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.grid()
    plt.legend()

    # 绘制对比损失
    plt.subplot(2, 2, 3)
    plt.plot(epoch_cl_loss_values, label="Contrastive Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Contrastive Loss")
    plt.grid()
    plt.legend()

    # 绘制重建损失
    plt.subplot(2, 2, 4)
    plt.plot(epoch_recon_loss_values, label="Recon Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Reconstruction Loss")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"loss_plot_epoch_{epoch + 1}.png"))
    plt.close()



def plot_dice(train_loss_values, val_dice_values, output_path):
    # Plot Training and Validation Metrics
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_values, label="Train Loss")
    plt.plot(val_dice_values, label="Validation Dice")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, "training_curve.png"))
    plt.close()


# 定义数据加载和预处理
def load_image_info(data_dir):
    # 图像文件路径
    img_paths = sorted(glob(os.path.join(data_dir, "*.nii.gz")))

    # 使用 LoadImaged 和 EnsureChannelFirstd 来预处理图像
    transforms = Compose([
        LoadImaged(keys=["image"]),         # 加载图像
        EnsureChannelFirstd(keys=["image"]) # 确保通道维度在第一维
    ])

    # 创建数据集和数据加载器
    data = [{"image": img} for img in img_paths]
    dataset = Dataset(data=data, transform=transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    return loader





def save_metrics_to_csv(train_loss_values, val_dice_values, num_epochs, output_path, filename="training_metrics.csv"):
    metrics_data = {
        "Epoch": list(range(1, num_epochs + 1)),  # 生成对应的 epoch 数
        "Train Loss": train_loss_values,
        "Validation Dice": val_dice_values,
    }
    metrics_df = pd.DataFrame(metrics_data)  # 将数据转换为 DataFrame
    csv_path = os.path.join(output_path, filename)  # 定义 CSV 文件路径
    metrics_df.to_csv(csv_path, index=False)  # 保存为 CSV 文件
    print(f"Training metrics saved to {csv_path}")


# 主程序
if __name__ == "__main__":
    data_dir = "/Users/xuan/Documents/01_study/CS_6357_open/project/FLARE/Training/FLARE22_LabeledCase/images"  # 替换为你的数据路径
    loader = load_image_info(data_dir)

    # 查看第一个图像的尺寸和像素值范围
    for batch in loader:
        image = batch["image"][0]  # 获取第一个图像样本
        print("图像尺寸:", image.shape)  # 打印图像尺寸
        print("像素值范围:", image.min().item(), "-", image.max().item())  # 打印像素值的最小值和最大值
        break  # 只打印第一个样本的尺寸和范围
