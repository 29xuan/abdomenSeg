import os
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord
from tqdm import tqdm
import nibabel as nib
from scipy.ndimage import zoom
from monai.transforms import Resize

from stage_one_two.src.load_data import get_unlabeled_data_loader, pseudo_transforms

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FLARE_dir = "/Volumes/Xuan_ExFAT/open_project/FLARE"
FLARE_dir = "F:/open_project/FLARE"
# unlabeled_data_dir = os.path.join(FLARE_dir, "Training", "FLARE22_UnlabeledCase")
unlabeled_data_dir = os.path.join(FLARE_dir, "Tuning", "images")
pseudo_label_dir = os.path.join(FLARE_dir, "Tuning", "pseudo")
os.makedirs(pseudo_label_dir, exist_ok=True)

# Model and weight loading configuration
model_path = os.path.join(FLARE_dir, "outputs_tuning", "best_metric_model.pth")
num_classes = 14

def load_model(device, num_classes, model_path):
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def restore_original_resolution(prediction, original_shape):
    original_shape_np = [int(s.item()) for s in original_shape]
    scale_factors = [orig / pred for orig, pred in zip(original_shape_np, prediction.shape)]
    print(f"Scale factors for resizing: {scale_factors}")
    restored_prediction = zoom(prediction, scale_factors, order=0)  # Nearest-neighbor interpolation for labels
    return restored_prediction

def generate_pseudo_labels(model, unlabeled_loader,pseudo_label_dir, device, threshold=0.8):
    print("Using device:", device)
    pseudo_labels = {}

    with torch.no_grad():
        for batch in tqdm(unlabeled_loader, desc="Generating Pseudo Labels"):
            input_image = batch["image"].to(device)
            batch_filenames = batch["image_meta_dict"]["filename_or_obj"]
            original_shape = batch["image_meta_dict"]["original_shape"] # 获取原始分辨率
            affine = batch["image_meta_dict"]["affine"].numpy()
            print(f"Batch: Processing {len(batch_filenames)} images")

            # 打印加载的图像形状
            print(f"Batch: Processing {len(batch_filenames)} images")
            print(f"Input image shape: {input_image.shape}")
            print(f"Original shape: {original_shape}")

            # Generate pseudo labels
            outputs = sliding_window_inference(input_image, (96, 96, 96), 4, model)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()  # Shape: [B, H, W, D]

            # 打印预测伪标签的形状
            print(f"Pseudo label shape: {predictions[0].shape}")

            # Process each image in the batch
            for i, file_path in enumerate(batch_filenames):
                base_name = os.path.basename(file_path).replace(".nii.gz", "")
                pseudo_label_path = os.path.join(pseudo_label_dir, f"{base_name}_pseudo.nii.gz")

                # test messages
                print("-" * 40)
                print(f"Prediction shape before resizing: {predictions[i].shape}")

                # Restore prediction to original resolution
                restored_prediction = restore_original_resolution(predictions[i], original_shape)
                print(f"Restored prediction shape: {restored_prediction.shape}")

                # Save predicted label as .nii.gz file
                if affine.shape == (1, 4, 4):
                    affine = affine.squeeze(0)

                nifti_image = nib.Nifti1Image(restored_prediction.astype(np.uint8), affine=affine)
                print(f"NIfTI image shape: {nifti_image.shape}")
                nib.save(nifti_image, pseudo_label_path)

                print(f"Saved pseudo label for {base_name} at {pseudo_label_path}")

                # Calculate confidence score
                confidence_score = float(torch.max(outputs[i].softmax(dim=0)).item())
                if confidence_score >= threshold:
                    pseudo_labels[pseudo_label_path] = confidence_score
                    print(f"Generated pseudo label for {base_name} with confidence {confidence_score:.4f}")

    return pseudo_labels

if __name__ == "__main__":
    # Initialize DataLoaders
    # unlabeled_loader = get_unlabeled_data_loader(unlabeled_data_dir, pseudo_transforms, batch_size=1, num_workers=0)
    unlabeled_loader = get_unlabeled_data_loader(unlabeled_data_dir, pseudo_transforms, batch_size=1, num_workers=0)
    print(f'Unlabeled loader finish!')

    # Load model
    model = load_model(device, num_classes, model_path)
    print(f'model loaded finish!')

    pseudo_labels = generate_pseudo_labels(model, unlabeled_loader, pseudo_label_dir, device, threshold=0.8)

    print(f'psesudo label finish!')

    # 将生成的伪标签和置信度保存到文件，供下个脚本使用
    torch.save(pseudo_labels, os.path.join(pseudo_label_dir, "pseudo_labels_confidence.pt"))