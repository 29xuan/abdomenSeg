import os
import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged,
    CropForegroundd, SpatialPadd, RandSpatialCropSamplesd, RandCoarseDropoutd,
    RandCoarseShuffled, ToTensord, CopyItemsd
)
from monai.data import Dataset, DataLoader




from monai.transforms import (
Compose,
LoadImaged,
RandAffined,
Resized,
ScaleIntensityRanged,
RandAdjustContrastd,
RandCropByPosNegLabeld,
MapTransform

)
from monai.utils import set_determinism, first

import numpy as np
from monai import transforms

class Copyd(MapTransform):
    def __init__(self, keys, new_key) -> None:
        super().__init__(keys)
        self.new_key = new_key

    def __call__(self, data):
        # start_time = time.time()
        for i, key in enumerate(self.keys):

            data[self.new_key[i]] = np.copy(data[key])
        # print(f"Copyd duration: {time.time() - start_time}s")

        return data
class Copypathd(MapTransform):
    def __init__(self, keys, new_key) -> None:
        super().__init__(keys)
        if isinstance(new_key, list):
            self.new_key = new_key
        else:
            self.new_key = [new_key] * len(keys)

    def __call__(self, data):
        for i, key in enumerate(self.keys):
            if isinstance(data[key], str):
                # If the data associated with the key is a file path (string), copy it to the new key
                data[self.new_key[i]] = data[key]
            else:
                raise TypeError(f"Expected a file path string for key '{key}', but got {type(data[key])}")

        return data




# Function to save NIfTI image
def save_nifti(data, filename, output_dir):
    numpy_data = data.squeeze(0).numpy()
    img = nib.Nifti1Image(numpy_data.astype(np.float32), np.eye(4))  # Using identity matrix for affine
    nib.save(img, os.path.join(output_dir, filename))
    print(f"Saved: {filename}")


# Define the transformations
self_sup_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
        RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96, 96), random_size=False, num_samples=2),
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),


        # Apply dropout and shuffle to the copied images
        RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32),
        # Copyd(keys=["image"], new_key="in_i"),
        CopyItemsd(keys=["image"], times=1, names=["in_i"], allow_missing_keys=False),
        RandCoarseShuffled(keys=["image"], prob=1, holes=30, spatial_size=20),
        # Copyd(keys=["image"], new_key="in_sh_i"),
        CopyItemsd(keys=["image"], times=1, names=["in_sh_i"], allow_missing_keys=False),





        RandCoarseDropoutd(keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64),
        # Copyd(keys=["image_2"], new_key="out_i"),
        CopyItemsd(keys=["image_2"], times=1, names=["out_i"], allow_missing_keys=False),
        RandCoarseShuffled(keys=["image_2"], prob=1, holes=30, spatial_size=20),
        # Copyd(keys=["image_2"], new_key="out_sh_i"),
        CopyItemsd(keys=["image_2"], times=1, names=["out_sh_i"], allow_missing_keys=False),
        #
        ToTensord(keys=["image", "gt_image", "image_2", "in_i", "in_sh_i", "out_i", "out_sh_i"])
    ]
)



self_sup_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3071, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
        RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96, 96), random_size=False, num_samples=2),


        # Apply dropout and shuffle to the copied images
        RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32),
        # Copyd(keys=["image"], new_key="in_i"),
        CopyItemsd(keys=["image"], times=1, names=["in_i"], allow_missing_keys=False),
        RandCoarseShuffled(keys=["image"], prob=1, holes=30, spatial_size=20),
        # Copyd(keys=["image"], new_key="in_sh_i"),
        CopyItemsd(keys=["image"], times=1, names=["in_sh_i"], allow_missing_keys=False),





        RandCoarseDropoutd(keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64),
        # Copyd(keys=["image_2"], new_key="out_i"),
        CopyItemsd(keys=["image_2"], times=1, names=["out_i"], allow_missing_keys=False),
        RandCoarseShuffled(keys=["image_2"], prob=1, holes=30, spatial_size=20),
        # Copyd(keys=["image_2"], new_key="out_sh_i"),
        CopyItemsd(keys=["image_2"], times=1, names=["out_sh_i"], allow_missing_keys=False),
        #
        ToTensord(keys=["image", "gt_image", "image_2", "in_i", "in_sh_i", "out_i", "out_sh_i"])
    ]
)

# Load the input image
image_path = "/Volumes/Xuan_ExFAT/open_project/FLARE/Training/FLARE22_UnlabeledCase/Case_00001_0000.nii.gz"
output_dir = "output2"
os.makedirs(output_dir, exist_ok=True)

# Input data dictionary
data_dict = {"image": image_path}

# Create the dataset and apply transformations
dataset = Dataset(data=[data_dict], transform=self_sup_transforms)

trainloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# Process the first batch
first_batch = first(trainloader)

# Extract the transformed images
gt = first_batch["gt_image"][0]

in_i = first_batch["in_i"][0]
in_sh_i = first_batch["in_sh_i"][0]
out_i = first_batch["out_i"][0]
out_sh_i = first_batch["out_sh_i"][0]

# Save the processed images
save_nifti(gt, "gt.nii.gz", output_dir)
save_nifti(in_i, "in_i.nii.gz", output_dir)
save_nifti(in_sh_i, "in_sh_i.nii.gz", output_dir)
save_nifti(out_i, "out_i.nii.gz", output_dir)
save_nifti(out_sh_i, "out_sh_i.nii.gz", output_dir)
