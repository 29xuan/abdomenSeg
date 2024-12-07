import os
import numpy as np
import torch
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
import monai
from glob import glob
from monai.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from monai.utils import set_determinism
import argparse

set_determinism(seed=42)

def plot_dice(train_loss_values, val_dice_values, output_path, args):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(train_loss_values, label="Train Loss", color="blue", linestyle="-")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Train Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    
    ax2 = ax1.twinx()
    ax2.plot(val_dice_values, label="Validation Dice", color="green", linestyle="-")
    ax2.set_ylabel("Validation Dice ", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # Add grid and legend
    ax1.grid()
    # fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    
    # Save the figure
    plt.legend()
    plt.savefig(os.path.join(output_path, "curve_{}_{}.png".format(args.flag, args.weight)))
    plt.close()


def save_metrics_to_csv(train_loss_values, val_dice_values, num_epochs, output_path, args):
    metrics_data = {
        "Train Loss": train_loss_values,
        "Validation Dice": val_dice_values,
    }
    metrics_df = pd.DataFrame(metrics_data)  # 将数据转换为 DataFrame
    csv_path = os.path.join(output_path, 'metrics_{}_{}.csv'.format(args.flag, args.weight))
    metrics_df.to_csv(csv_path, index=False)  # 保存为 CSV 文件
    print(f"Training metrics saved to {csv_path}")


from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged, CropForegroundd,
    SpatialPadd, RandSpatialCropSamplesd, CopyItemsd, OneOf, RandCoarseDropoutd,
    RandCoarseShuffled, Compose, ToTensord, ScaleIntensityd, Resized, Orientationd, RandCropByPosNegLabeld, RandFlipd,
    RandRotate90d, RandShiftIntensityd
)

train_labeled_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Reorient images and labels to conform to RAS (right-front-top)
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Adjust the voxel spacing of images and labels. Images use bilinear interpolation, while labels use nearest neighbor interpolation.
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest"),),
        # Scaling based on the actual pixel value range of the image
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0.0, b_max=1.0, clip=True),
        # Crop image foreground
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Make sure the image size meets the requirement after cropping
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode="constant"),
        # Padding to at least (96, 96, 96)
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode="constant"),
        # Randomly crop a (96, 96, 96) volume block, ensuring that the cropped area contains some foreground (pos=1 means that the cropped block contains at least one foreground voxel),
        # and neg=1 means that some background voxels are allowed. Crop 4 samples per image
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0),
        # Randomly flip images and labels along 3 spatial axes
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        # Randomly rotate images and labels by 90 degrees
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        # Randomly increase or decrease the pixel intensity of an image by up to 10%
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        ToTensord(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

def get_roi_data_loader(image_dir, label_dir, labeled_transforms=train_labeled_transforms, batch_size=2, num_workers=0, shuffle=True):
    # Get the image and label paths of the labeled data
    # the intermediate results are saved in 
    # ROIs_FLARE22_labeledCase_aug/   ROIs_FLARE22_labeledCase_image/ ROIs_FLARE22_labeledCase_label/
    img_paths = sorted(glob(os.path.join(image_dir, "*.nii.gz")))
    label_paths = sorted(glob(os.path.join(label_dir, "*.nii.gz")))
    print('Number of images:', len(img_paths))
    print('Number of labels:', len(label_paths))
    # Define Data Transformations
    labeled_transforms = labeled_transforms

    # Create a labeled dataset
    train_files = [{"image": img, "label": label} for img, label in zip(img_paths, label_paths)]
    ds = Dataset(data=train_files, transform=labeled_transforms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def get_mixed_roi_data_loader(image_dir, label_dir, \
                        pseudo_image_dir, pseudo_label_dir, sample_num, \
                        labeled_transforms=train_labeled_transforms, batch_size=2, num_workers=0, shuffle=True):
    img_paths = sorted(glob(os.path.join(image_dir, "*.nii.gz")))
    label_paths = sorted(glob(os.path.join(label_dir, "*.nii.gz")))
    pseudo_img_paths = sorted(glob(os.path.join(pseudo_image_dir, "*.nii.gz")))
    pseudo_label_paths = sorted(glob(os.path.join(pseudo_label_dir, "*.nii.gz")))
    print('Number of images:', len(img_paths))
    print('Number of labels:', len(label_paths))
    print('Number of pseudo images:', len(pseudo_img_paths))
    print('Number of pseudo labels:', len(pseudo_label_paths))
    # Define Data Transformations
    labeled_transforms = labeled_transforms

    # Create a labeled dataset
    train_files = [{"image": img, "label": label} for img, label in zip(img_paths, label_paths)]
    indices = np.random.choice(len(pseudo_img_paths), sample_num, replace=False)
    pseudo_img_paths = [pseudo_img_paths[i] for i in indices]
    pseudo_label_paths = [pseudo_label_paths[i] for i in indices]
    train_files_pseudo = [{"image": img, "label": label} for img, label in zip(pseudo_img_paths, pseudo_label_paths)]
    train_files.extend(train_files_pseudo)
    ds = Dataset(data=train_files, transform=labeled_transforms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLARE_dir = "/home/yuying/abdomen_seg/FLARE"
output_path = "./outputs_tuning"
os.makedirs(output_path, exist_ok=True)
num_classes=14

# Define Loss, Optimizer, and Metrics
loss_function = DiceCELoss(to_onehot_y=True, softmax=True).to(device)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_label = AsDiscrete(to_onehot=num_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

# Network Configuration
def initialize_model(device, num_classes=14):
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    return model


# Configuring training loop parameters
num_epochs = 50
global_step = 0
lr = 1e-4
early_stop_cnt = 0

# Training and Validation Functions
def train(model, train_loader, pseudo_loader, optimizer, loss_function, device):
    print("Using device:", device)
    model.train()
    epoch_loss = 0
    steps = 0

    for batch, pseudo_batch in zip(train_loader, pseudo_loader):
        steps += 1
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        outputs = model(inputs)
        # print(f"Outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
        # print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
        # print(f"Labels unique values: {torch.unique(labels)}")
        loss = loss_function(outputs, labels)

        pseudo_inputs, pseudo_labels = pseudo_batch["image"].to(device), pseudo_batch["label"].to(device)
        pseudo_outputs = model(pseudo_inputs)
        pseudo_loss = loss_function(pseudo_outputs, pseudo_labels)
        loss += args.weight * pseudo_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / steps

def validate(model, val_loader, dice_metric, post_label, post_pred, device):
    model.eval()
    dice_vals = []

    with torch.no_grad():
        for batch in val_loader:
            val_inputs, val_labels = batch["image"].to(device), batch["label"].to(device)
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label) for val_label in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred) for val_pred in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice_vals.append(dice_metric.aggregate().item())
            dice_metric.reset()

    return np.mean(dice_vals)

def run_finetuning(args, model, train_loader, pseudo_loader, val_loader, optimizer, loss_function, dice_metric, post_label, post_pred, output_path, device, num_epochs=50, eval_interval=5):
    best_dice = 0.0
    train_loss_values = []
    val_dice_values = []

    for epoch in range(num_epochs):
        print("-" * 20)
        print(f"Epoch {epoch + 1}/{num_epochs}")

        epoch_loss = train(model, train_loader, pseudo_loader, optimizer, loss_function, device)
        train_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # validate
        val_dice = validate(model, val_loader, dice_metric, post_label, post_pred, device)
        val_dice_values.append(val_dice)
        print(f"Epoch {epoch + 1} average Dice Score: {val_dice:.4f}")

        # Save the best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(output_path, f"best_model_{args.flag}_{args.weight}.pth"))
            print(f"Saved Best Model at Epoch: {epoch+1} with Dice Score: {best_dice:.4f}")
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
            if early_stop_cnt == args.patience:
                print(f"Early stopping at epoch: {epoch+1}")
                break
        if (epoch + 1) % eval_interval == 0:
            plot_dice(train_loss_values, val_dice_values, output_path, args)

    # save final midel
    torch.save(model.state_dict(), os.path.join(output_path, f"final_model_{args.flag}_{args.weight}.pth"))
    print(f"Training complete with best validation dice: {best_dice:.4f}. Final model saved.")

    save_metrics_to_csv(train_loss_values, val_dice_values, num_epochs, output_path, args)

# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", type=str, default='label', help="Flag for running fine-tuning")
    parser.add_argument("--weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    print(f"Running Fine-Tuning with flag: {args.flag}, {args.weight}, {args.patience}")


    # Initialize DataLoaders
    FLARE_dir = "/home/yuying/abdomen_seg/FLARE"
    # labeled case
    labeled_roi_image_data_dir = os.path.join(FLARE_dir, "Training", "ROIs_FLARE22_labeledCase_image")
    labeled_roi_label_data_dir = os.path.join(FLARE_dir, "Training", "ROIs_FLARE22_labeledCase_label")
    labeled_roi_aug_image_data_dir = os.path.join(FLARE_dir, "Training", "ROIs_FLARE22_labeledCase_aug")
    # unlabeled case
    pseudo_roi_image_data_dir = os.path.join(FLARE_dir, "Training", "ROIs_FLARE22_unlabeledCase_image")
    pseudo_roi_label_data_dir = os.path.join(FLARE_dir, "Training", "ROIs_FLARE22_unlabeledCase_label")
    pseudo_roi_aug_image_data_dir = os.path.join(FLARE_dir, "Training", "ROIs_FLARE22_unlabeledCase_aug")
    # labeled validation case
    labeled_roi_val_image_data_dir = os.path.join(FLARE_dir, "Tuning", "ROIs_FLARE22_labeledCase_image")
    labeled_roi_val_label_data_dir = os.path.join(FLARE_dir, "Tuning", "ROIs_FLARE22_labeledCase_label")

    if args.weight == 0:
        exit(0)

    train_loader = get_roi_data_loader(labeled_roi_image_data_dir, labeled_roi_label_data_dir, train_labeled_transforms, batch_size=1)
    pseudo_loader = get_roi_data_loader(pseudo_roi_image_data_dir, pseudo_roi_label_data_dir, train_labeled_transforms, batch_size=1)

    val_loader = get_roi_data_loader(labeled_roi_val_image_data_dir, labeled_roi_val_label_data_dir, val_transforms, batch_size=1)
    
    # Initialize Model
    model = initialize_model(device, num_classes=num_classes)

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Run Fine-Tuning
    run_finetuning(args, model, train_loader, pseudo_loader, val_loader, optimizer, loss_function, dice_metric, post_label, post_pred, output_path, device, num_epochs=1000, eval_interval=3)




