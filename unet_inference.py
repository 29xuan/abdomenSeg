# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from glob import glob
import torch
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
import argparse
import SimpleITK as sitk
import numpy as np

from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    Resized,
    SaveImaged,
    ScaleIntensityd,
)

from monai.utils import set_determinism
set_determinism(seed=42)


def get_bounding_box(label_image, padding=5):
    label_array = sitk.GetArrayFromImage(label_image)
    non_zero_coords = np.argwhere(label_array > 0)

    if len(non_zero_coords) == 0:
        return None

    # Convert shape to numpy array for calculations
    label_shape = np.array(label_array.shape)

    # Get min and max coordinates along each axis (Z, Y, X)
    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0)

    # Adjust for padding
    min_coords = np.maximum(min_coords - padding, 0)
    max_coords = np.minimum(max_coords + padding, label_shape - 1)

    # Convert to (X, Y, Z) order for SimpleITK
    min_coords = min_coords[::-1]
    max_coords = max_coords[::-1]

    return min_coords, max_coords


def recover_image(input_image_path, input_label_path, \
                  roi_label_tensor_image, \
                  output_label_path, save_flag=True):
    image = sitk.ReadImage(input_image_path)
    label_image = sitk.ReadImage(input_label_path)

    bounding_box = get_bounding_box(label_image)
    if bounding_box is None:
        print(f"No labels found in {input_label_path}")
        return

    min_coords, max_coords = bounding_box

    # Ensure the region is within bounds
    image_shape = np.array(image.GetSize())
    if np.any(min_coords < 0) or np.any(max_coords >= image_shape):
        print(f"Skipping {input_image_path}: Region out of bounds.")
        return

    # Define ROI size and start
    size = max_coords - min_coords + 1
    start = min_coords

    transposed_tensor = np.transpose(roi_label_tensor_image, (2, 1, 0)) # this is to solve the mismatch between SimpleITK and PyTorch
    roi_label_image = sitk.GetImageFromArray(transposed_tensor)
    roi_label_image = sitk.Cast(roi_label_image, label_image.GetPixelID())
    # print('Tensor image size:', roi_label_tensor_image.shape)
    # print("Image size (ori image, roi image):", image.GetSize(), roi_label_image.GetSize())

    pasted_image = sitk.Paste(label_image, roi_label_image, size.tolist(), destinationIndex=start.tolist(), sourceIndex=[0, 0, 0])

    # print("Image size:", image.GetSize(), roi_label_image.GetSize(), pasted_image.GetSize())

    if save_flag:
        sitk.WriteImage(pasted_image, output_label_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", type=str, default='label', help="Flag for running fine-tuning")
    parser.add_argument("--weight", type=float, default=0.0)
    args = parser.parse_args()
    print(f"Running Fine-Tuning with flag: {args.flag}, {args.weight}")

    FLARE_dir = "/home/yuying/abdomen_seg/FLARE"
    test_image_data_dir = os.path.join(FLARE_dir, "Testing", "images")
    test_label_data_dir = os.path.join(FLARE_dir, "Testing", "labels")
    test_roi_image_data_dir = os.path.join(FLARE_dir, "Testing", "ROIs_FLARE22_labeledCase_image")
    output_dir = "./outputs_tuning/predictions_{}_{}".format(args.flag, args.weight)
    os.makedirs(output_dir, exist_ok=True)

    test_roi_images = sorted(glob(os.path.join(test_roi_image_data_dir, "*.nii.gz")))
    test_files = [{"img": img} for img in test_roi_images]

    # define pre transforms
    pre_transforms = Compose(
        [
            LoadImaged(keys="img"),
            EnsureChannelFirstd(keys="img"),
            Orientationd(keys="img", axcodes="RAS"),
            Resized(keys="img", spatial_size=(96, 96, 96), mode="trilinear", align_corners=True),
            ScaleIntensityd(keys="img"),
        ]
    )
    # define dataset and dataloader
    test_dataset = Dataset(data=test_files, transform=pre_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)
    # define post transforms
    post_transforms = Compose(
    [
        Activationsd(keys="pred", softmax=True),  # Use Softmax for multi-class outputs
        Invertd(
            keys="pred",  # Invert the `pred` data field
            transform=pre_transforms,
            orig_keys="img",  # Use the pre-transforms info from `img` field
            nearest_interp=False,  # Ensure smooth output
            to_tensor=True,  # Convert to PyTorch Tensor after inverting
        ),
        AsDiscreted(keys="pred", argmax=True),  # Use `argmax` for discretization in multi-class
        # SaveImaged(keys="pred", output_dir=output_dir, 
        #            output_postfix="", resample=False,
        #            separate_folder=False),
    ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=14,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model_file_name = os.path.join("./outputs_tuning", "best_model_{}_{}.pth".format(args.flag, args.weight))
    net.load_state_dict(torch.load(model_file_name))

    net.eval()
    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            images = d["img"].to(device)
            d["pred"] = sliding_window_inference(inputs=images, roi_size=(96, 96, 96), sw_batch_size=4, predictor=net)
            d = [post_transforms(i) for i in decollate_batch(d)]
            roi_prediction_labels = d[0]["pred"][0] # 3d dimensions
            roi_prediction_labels = roi_prediction_labels.cpu().numpy()
            # print(roi_prediction_labels.shape)

            # recover the roi into the original image
            test_filename = os.path.basename(test_roi_images[i])
            input_image_path = os.path.join(test_image_data_dir, test_filename)
            input_label_path = os.path.join(test_label_data_dir, test_filename).replace("_0000", "")
            output_path = os.path.join(output_dir, test_filename)
            # print(input_image_path, input_label_path, output_path)
            recover_image(input_image_path, input_label_path, \
                  roi_prediction_labels, \
                  output_path, save_flag=True)
    print('Finish', args.flag, args.weight)

if __name__ == "__main__":
    main()