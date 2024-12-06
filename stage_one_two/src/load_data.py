import os
from glob import glob
import monai
import nibabel as nib
import numpy as np
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged, CropForegroundd,
    SpatialPadd, RandSpatialCropSamplesd, CopyItemsd, OneOf, RandCoarseDropoutd,
    RandCoarseShuffled, Compose, ToTensord, ScaleIntensityd, Resized, Orientationd, RandCropByPosNegLabeld, RandFlipd,
    RandRotate90d, RandShiftIntensityd
)
from monai.data import Dataset, DataLoader


self_sup_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            # Adjust voxel size to match actual data
            Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
            # Scaling based on the actual pixel value range of the image
            ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0.0, b_max=1.0, clip=True),
            # Crop image foreground
            CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
            # Fill the image to the specified size
            SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
            # Randomly crop samples
            RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96, 96), random_size=False, num_samples=2),
            # Comparison of Copying Images for Self-Supervised Learning
            CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
            # Randomly perform different hole drop enhancements
            OneOf(transforms=[
                # inner-cutout
                RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32),
                # outer-cutout
                RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64),
            ]),
            # Random coarse-grained shuffled pixel blocks
            RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
            # Apply different enhancements to the image of the second view
            OneOf(transforms=[
                RandCoarseDropoutd(keys=["image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32),
                RandCoarseDropoutd(keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64),
            ]),
            RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
            ToTensord(keys=["image", "image_2", "gt_image"]),
        ]
    )


# Transforms
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


class CustomLoadImaged(LoadImaged):
    def __call__(self, data):
        result = super().__call__(data)

        # Manually load affine information from the original file
        file_path = data["image"]
        try:
            nib_image = nib.load(file_path)  # Loading image files using nibabel
            affine = nib_image.affine  # Extracting the affine matrix
            original_shape = nib_image.shape  # Extract the original shape of the image
        except Exception as e:
            print(f"Error loading affine information for {file_path}: {e}")
            affine = np.eye(4)  # If loading fails, use the identity matrix as the default value
            original_shape = None  # If loading fails, the shape is set to None

        # Add to meta information
        result["image_meta_dict"] = {
            "filename_or_obj": file_path,
            "affine": affine,
            "original_shape": original_shape
        }
        return result

pseudo_transforms = Compose([
        CustomLoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image"]),
    ])


def get_labeled_data_loader(labeled_data_dir, labeled_transforms=train_labeled_transforms, batch_size=2, num_workers=0):
    # Get the image and label paths of the labeled data
    img_paths = sorted(glob(os.path.join(labeled_data_dir, "images", "*.nii.gz")))
    label_paths = sorted(glob(os.path.join(labeled_data_dir, "labels", "*.nii.gz")))

    # Define Data Transformations
    labeled_transforms = labeled_transforms

    # Create a labeled dataset
    train_files = [{"image": img, "label": label} for img, label in zip(img_paths, label_paths)]
    ds = Dataset(data=train_files, transform=labeled_transforms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return loader

def get_unlabeled_data_loader(unlabeled_data_dir, unlabeled_transforms=self_sup_transforms, batch_size=4, num_workers=4):
    # Get the image and label paths of the labeled data
    unlabeled_img_paths = sorted(glob(os.path.join(unlabeled_data_dir, "*.nii.gz")))

    valid_img_paths = [img for img in unlabeled_img_paths if not os.path.basename(img).startswith(".")]

    # Define Data Transformations
    unlabeled_transforms = unlabeled_transforms

    # Create a unlabeled dataset
    unlabeled_files = [{"image": img} for img in valid_img_paths]
    unlabeled_ds = Dataset(data=unlabeled_files, transform=unlabeled_transforms)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return unlabeled_loader



if __name__ == '__main__':
    # FLARE_dir = "/Users/xuan/Documents/01_study/CS_6357_open/project/FLARE"
    FLARE_dir = "F:/open_project/FLARE"
    labeled_data_dir = os.path.join(FLARE_dir, "Training", "FLARE22_LabeledCase")
    unlabeled_data_dir = os.path.join(FLARE_dir, "Training", "FLARE22_UnlabeledCase")
    validation_data_dir = os.path.join(FLARE_dir, "Tuning")


    class CustomLoadImaged(LoadImaged):
        def __call__(self, data):
            result = super().__call__(data)
            # Manually add meta information
            result["image_meta_dict"] = {"filename_or_obj": data["image"]}
            return result


    # Using a custom Loader
    loader = CustomLoadImaged(keys=["image"])
    test_image = os.path.join(unlabeled_data_dir, "Case_00001_0000.nii.gz")
    data = loader({"image": test_image})

    print(data.keys())
    if "image_meta_dict" in data:
        print(f"Meta keys: {data['image_meta_dict']}")
    else:
        print("image_meta_dict is missing.")

    img = nib.load(test_image)
    print(img.header)
    #
    # # Get the data loader
    # labeled_loader = get_labeled_data_loader(labeled_data_dir, train_labeled_transforms, 2)
    # unlabeled_loader = get_unlabeled_data_loader(unlabeled_data_dir, self_sup_transforms, 2)
    # val_loader = get_labeled_data_loader(validation_data_dir, val_transforms, 1)
    #
    #
    # # Print the shape of some sample data
    # print(f"Number of labeled batches: {len(labeled_loader)}")
    # for batch in labeled_loader:
    #     print("Labeled data batch:")
    #     print(f"Image shape: {batch['image'].shape}")
    #     print(f"Label shape: {batch['label'].shape}")
    #     break  # Print only the first batch to verify that the load was successful
    #
    # print(f"Number of unlabeled batches: {len(labeled_loader)}")
    # for batch in unlabeled_loader:
    #     print("Unlabeled data batch:")
    #     print(f"Image shape: {batch['image'].shape}")
    #     break
    #
    # print(f"Number of validation batches: {len(val_loader)}")
    # for batch in val_loader:
    #     print("Validation data batch:")
    #     print(f"Image shape: {batch['image'].shape}")
    #     print(f"Label shape: {batch['label'].shape}")
    #     break

