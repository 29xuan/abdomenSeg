import os
import numpy as np
import torch
from tqdm import tqdm
from monai.networks.nets import UNETR
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from load_data import get_labeled_data_loader, train_labeled_transforms, val_transforms
from stage_one_two.utils import plot_dice, save_metrics_to_csv
import torchvision
import warnings

import torch



# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_pretrained = True
# FLARE_dir = "/Users/xuan/Documents/01_study/CS_6357_open/project/FLARE"
FLARE_dir = "F:/open_project/FLARE"
pretrained_path = os.path.join(FLARE_dir, "outputs_self", "best_model.pth")
labeled_train_data_dir = os.path.join(FLARE_dir, "Training", "FLARE22_labeledCase")
labeled_val_data_dir = os.path.join(FLARE_dir, "Tuning")
output_path = os.path.join(FLARE_dir, "outputs_tuning")
os.makedirs(output_path, exist_ok=True)
num_classes=14

# Define Loss, Optimizer, and Metrics
loss_function = DiceCELoss(to_onehot_y=True, softmax=True).to(device)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_label = AsDiscrete(to_onehot=num_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

# Network Configuration
def initialize_model(device, num_classes=14, pretrained_path=None):
    model = UNETR(
        in_channels=1,
        out_channels=num_classes,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="conv",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)

    if pretrained_path:
        print(f"Loading pre-trained ViT backbone weights from {pretrained_path}")
        vit_dict = torch.load(pretrained_path, map_location=device)
        vit_weights = vit_dict["state_dict"] if "state_dict" in vit_dict else vit_dict

        model_dict = model.vit.state_dict()
        vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
        model_dict.update(vit_weights)
        model.vit.load_state_dict(model_dict)
        print("Pretrained Weights Successfully Loaded!")
    else:
        print("No weights loaded; using randomly initialized weights.")

    return model


# Configuring training loop parameters
num_epochs = 50
global_step = 0
lr = 1e-4

# Training and Validation Functions
def train(model, train_loader, optimizer, loss_function, device):
    print("Using device:", device)
    model.train()
    epoch_loss = 0
    steps = 0

    for batch in train_loader:
        steps += 1
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        outputs = model(inputs)
        # print(f"Outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
        # print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
        # print(f"Labels unique values: {torch.unique(labels)}")
        loss = loss_function(outputs, labels)

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

def run_finetuning(model, train_loader, val_loader, optimizer, loss_function, dice_metric, post_label, post_pred, output_path, device, num_epochs=50, eval_interval=5):
    best_dice = 0.0
    train_loss_values = []
    val_dice_values = []

    for epoch in range(num_epochs):
        print("-" * 20)
        print(f"Epoch {epoch + 1}/{num_epochs}")

        epoch_loss = train(model, train_loader, optimizer, loss_function, device)
        train_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # validate
        val_dice = validate(model, val_loader, dice_metric, post_label, post_pred, device)
        val_dice_values.append(val_dice)
        print(f"Epoch {epoch + 1} average Dice Score: {val_dice:.4f}")

        # Save the best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(output_path, f"best_metric_model.pth"))
            print(f"Saved Best Model at Epoch: {epoch+1} with Dice Score: {best_dice:.4f}")

        if (epoch + 1) % eval_interval == 0:
            plot_dice(train_loss_values, val_dice_values, output_path)

    # save final midel
    torch.save(model.state_dict(), os.path.join(output_path, f"final_model_{epoch+1}.pth"))
    print(f"Training complete with best validation dice: {best_dice:.4f}. Final model saved.")

    # 调用保存函数
    save_metrics_to_csv(train_loss_values, val_dice_values, num_epochs, output_path)

# Main Execution
if __name__ == "__main__":
    # print(torch.__config__.show())
    torch.cuda.empty_cache()
    # Initialize DataLoaders
    train_loader = get_labeled_data_loader(labeled_train_data_dir, train_labeled_transforms, batch_size=1, num_workers=1)
    val_loader = get_labeled_data_loader(labeled_val_data_dir, val_transforms, batch_size=1, num_workers=1)

    # Initialize Model
    model = initialize_model(device, num_classes=num_classes, pretrained_path=pretrained_path)

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Run Fine-Tuning
    run_finetuning(model, train_loader, val_loader, optimizer, loss_function, dice_metric, post_label, post_pred, output_path, device, num_epochs=1000, eval_interval=3)




