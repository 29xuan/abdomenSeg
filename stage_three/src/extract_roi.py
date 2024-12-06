import SimpleITK as sitk
import os
import numpy as np


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

# Contrast enhancement and histogram equalization enhancement
def enhance_contrast(image, alpha=0.3, beta = 0.3):
    filter = sitk.AdaptiveHistogramEqualizationImageFilter()
    filter.SetAlpha(alpha)  # contrast enhancement degree
    filter.SetBeta(beta)  # Brightness retention
    enhanced_image = filter.Execute(image)

    return enhanced_image

# Gaussian blur
def apply_gaussian_blur(image, sigma=1.0):
    return sitk.SmoothingRecursiveGaussian(image, sigma)

# Gamma correction
def apply_gamma_correction(image, gamma=1.0):
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    array = ((array / np.max(array)) ** gamma) * 255
    array = np.clip(array, 0, 255).astype(np.uint8)
    corrected_image = sitk.GetImageFromArray(array)
    corrected_image.CopyInformation(image)
    return corrected_image

# Enhance the ROI area.
def augment_roi(image, methods=None):

    if not methods:
        return [image]

    augmented_image = image

    for method in methods:
        if method == "contrast":
            augmented_image = enhance_contrast(augmented_image)

        elif method == "gaussian":
            augmented_image = apply_gaussian_blur(augmented_image)

        elif method == "gamma":
            augmented_image = apply_gamma_correction(augmented_image)

    return augmented_image

# one method one augment image
def augment_roi_list(image, methods=None):

    if not methods:
        return [image]

    augmented_images = []

    if "contrast" in methods:
        augmented_images.append(enhance_contrast(image))

    if "gaussian" in methods:
        augmented_images.append(apply_gaussian_blur(image, sigma=1.0))

    if "gamma" in methods:
        augmented_images.append(apply_gamma_correction(image, gamma=1.2))

    return augmented_images

def extract_and_save_images(input_image_path, input_label_path, output_image_path, output_label_path, augmentation_output_path=None, augmentation_methods=None):
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

    # Crop the image and label
    cropped_image = sitk.RegionOfInterest(image, size.tolist(), start.tolist())
    cropped_label_image = sitk.RegionOfInterest(label_image, size.tolist(), start.tolist())

    # Save original cropped image and label
    sitk.WriteImage(cropped_image, os.path.join(output_image_path, os.path.basename(input_image_path)))
    sitk.WriteImage(cropped_label_image, os.path.join(output_label_path, os.path.basename(input_label_path)))

    print(f"Saved cropped image and label for {input_image_path}")

    # Apply augmentation to the cropped ROI if methods are provided
    if augmentation_methods:
        if not os.path.exists(augmentation_output_path):
            os.makedirs(augmentation_output_path)

        augmented_roi = augment_roi(cropped_image, methods=augmentation_methods)
        augmented_image_path = os.path.join(augmentation_output_path,
                                            f"{os.path.basename(input_image_path).replace('.nii.gz', '')}_aug.nii.gz")
        sitk.WriteImage(augmented_roi, augmented_image_path)
        print(f"Saved augmented ROI image to {augmented_image_path}")
        # augmentation_rois_list = augment_roi_list(cropped_image, methods=augmentation_methods)
        # for idx, augmented_roi in enumerate(augmented_rois):
        #     augmented_image_path = os.path.join(augmentation_output_path, f"{os.path.basename(input_image_path).replace('.nii.gz', '')}_aug_{i}.nii.gz")
        #     sitk.WriteImage(augmented_roi, augmented_image_path)
        #     print(f"Saved augmented ROI image to {augmented_image_path}")
    else:
        print(f"No augmentation applied for {input_image_path}")



def process_images(image_folder, label_folder, output_image_folder, output_label_folder, augmentation_output_folder=None, is_pseudo_label=False, augmentation_methods=None):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)

    if augmentation_methods and augmentation_output_folder and not os.path.exists(augmentation_output_folder):
        os.makedirs(augmentation_output_folder)

    image_path = os.listdir(image_folder)
    valid_img_paths = [img for img in image_path if not os.path.basename(img).startswith(".")]

    # Iterate through the image and label folders
    for image_name in valid_img_paths:
        if image_name.endswith(".nii.gz"):
            print("image name :", image_name)
            label_name = image_name.replace(".nii.gz", "_pseudo.nii.gz") if is_pseudo_label else image_name.replace("_0000.nii.gz", ".nii.gz")
            print("label name:", label_name)
            image_path = os.path.join(image_folder, image_name)
            label_path = os.path.join(label_folder, label_name)

            # Process each image and label pair
            if os.path.exists(label_path):
                extract_and_save_images(image_path, label_path, output_image_folder, output_label_folder, augmentation_output_folder, augmentation_methods)
            else:
                print(f"Label image {label_name} not found for {image_name}")



# FLARE_dir = "F:/open_project/FLARE"
FLARE_dir = "/Volumes/Xuan_ExFAT/open_project/FLARE"

# unlabeled data
# unlabeled_data_dir = os.path.join(FLARE_dir, "Training", "FLARE22_UnlabeledCase")
# pseudo_label_dir = os.path.join(FLARE_dir, "Training", "FLARE22_UnlabeledCase_Pseudo_Labels")
# roi_pseudo_dir = os.path.join(FLARE_dir, "Training", "ROIs_FLARE22_unlabeledCase_image")
# extract_pseudo_roi_label_dir = os.path.join(FLARE_dir,"Training", "ROIs_FLARE22_unlabeledCase_label")
# augmentation_pseudo_roi_dir = os.path.join(FLARE_dir, "Training", "ROIs_FLARE22_unlabeledCase_aug")
#
# # labeled data
# images_of_labeled_train_data_dir = os.path.join(FLARE_dir, "Training", "FLARE22_labeledCase", "images")
# labels_of_labeled_train_data_dir = os.path.join(FLARE_dir, "Training", "FLARE22_labeledCase", "labels")
# roi_labeled_dir = os.path.join(FLARE_dir, "Training", "ROIs_FLARE22_labeledCase_image")
# extract_labeled_roi_label_dir = os.path.join(FLARE_dir,"Training", "ROIs_FLARE22_labeledCase_label")
# augmentation_labeled_roi_dir = os.path.join(FLARE_dir,"Training", "ROIs_FLARE22_labeledCase_aug")

# # Tuning
# tuning_data_dir = os.path.join(FLARE_dir, "Tuning", "images")
# pseudo_label_dir = os.path.join(FLARE_dir, "Tuning", "pseudo")
# roi_pseudo_dir = os.path.join(FLARE_dir, "Tuning2", "ROIs_pseudo_image")
# extract_pseudo_roi_label_dir = os.path.join(FLARE_dir, "Tuning2", "ROIs_pseudo_label")
# augmentation_pseudo_roi_dir = os.path.join(FLARE_dir, "Tuning2", "ROIs_pseudo_aug")

# # labeled data
images_of_labeled_train_data_dir = os.path.join(FLARE_dir, "Tuning", "images")
labels_of_labeled_train_data_dir = os.path.join(FLARE_dir, "Tuning", "labels")
roi_labeled_dir = os.path.join(FLARE_dir, "Tunin3", "ROIs_labeled_image")
extract_labeled_roi_label_dir = os.path.join(FLARE_dir, "Tuning3", "ROIs_labeled_label")
augmentation_labeled_roi_dir = os.path.join(FLARE_dir, "Tuning3", "ROIs_labeled_aug")

# set augmentation_method
augmentation_methods = ["contrast", "gaussian", "gamma"]
# augmentation_methods = None  # or disable enhancements

# Processing labeled data and applying augmentation
# process_images(images_of_labeled_train_data_dir, labels_of_labeled_train_data_dir, roi_labeled_dir, extract_labeled_roi_label_dir, augmentation_labeled_roi_dir, is_pseudo_label=False, augmentation_methods=augmentation_methods)


# Processing unlabeled data and applying augmentation
# process_images(unlabeled_data_dir, pseudo_label_dir, roi_pseudo_dir, extract_pseudo_roi_label_dir, augmentation_pseudo_roi_dir, is_pseudo_label=True, augmentation_methods=augmentation_methods)


# process_images(tuning_data_dir, pseudo_label_dir, roi_pseudo_dir, extract_pseudo_roi_label_dir, augmentation_pseudo_roi_dir, is_pseudo_label=True, augmentation_methods=augmentation_methods)

process_images(images_of_labeled_train_data_dir, labels_of_labeled_train_data_dir, roi_labeled_dir, extract_labeled_roi_label_dir, augmentation_labeled_roi_dir, is_pseudo_label=False, augmentation_methods=augmentation_methods)
