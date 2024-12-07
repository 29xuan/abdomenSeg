import os
import nibabel as nib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
from sklearn.metrics import precision_score, recall_score
import argparse


def load_nifti_image(file_path):
    img = nib.load(file_path)
    return img.get_fdata()


import numpy as np
import os
import nibabel as nib


def calculate_metrics_3d(ground_truth, prediction, num_classes):
    """
    Calculate Dice Similarity Coefficient (DSC), IoU, Precision, Recall for each label in a 3D image.
    """
    metrics = {}
    for label in range(num_classes):
        # 计算每个标签的指标
        gt_mask = (ground_truth == label).astype(int)
        pred_mask = (prediction == label).astype(int)

        # 计算预测和真实标签的交集和并集
        intersection = np.sum(gt_mask * pred_mask)
        union = np.sum(gt_mask) + np.sum(pred_mask)
        true_positive = intersection
        false_positive = np.sum(pred_mask) - intersection
        false_negative = np.sum(gt_mask) - intersection

        # 计算各项指标，避免除零错误
        if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
            # 如果 ground truth 和预测都没有该标签，则跳过该标签的计算，设置为0
            dsc = iou = precision = recall = 0
        else:
            dsc = 2 * intersection / (np.sum(gt_mask) + np.sum(pred_mask)) if union != 0 else 0
            iou = intersection / union if union != 0 else 0
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0

        metrics[label] = {
            'DSC': dsc,
            'IoU': iou,
            'Precision': precision,
            'Recall': recall
        }
    return metrics


def process_folder(ground_truth_folder, predicted_folder, num_classes=14, output_file='metrics.txt'):
    """
    Process all NIfTI files in the given folders and calculate metrics for each class.
    """
    # 获取有效的 NIfTI 文件
    valid_gt_files = sorted([
        os.path.join(ground_truth_folder, f)
        for f in os.listdir(ground_truth_folder)
        if f.endswith('.nii.gz') and not f.startswith('.') and os.path.isfile(os.path.join(ground_truth_folder, f))
    ])
    valid_pred_files = sorted([
        os.path.join(predicted_folder, f)
        for f in os.listdir(predicted_folder)
        if f.endswith('.nii.gz') and not f.startswith('.') and os.path.isfile(os.path.join(predicted_folder, f))
    ])

    if len(valid_pred_files) != len(valid_gt_files):
        raise ValueError("Ground truth and prediction directories must contain the same number of files")

    all_metrics = {i: {'DSC': [], 'IoU': [], 'Precision': [], 'Recall': []} for i in range(num_classes)}

    # 遍历每个文件，计算指标
    for gt_file, pred_file in zip(valid_gt_files, valid_pred_files):
        print(f"Processing: {gt_file} vs {pred_file}")

        # 加载 NIfTI 文件
        ground_truth = load_nifti_image(gt_file)
        prediction = load_nifti_image(pred_file)

        # 计算指标
        metrics = calculate_metrics_3d(ground_truth, prediction, num_classes)

        # 合并每个类的指标
        for label, label_metrics in metrics.items():
            for metric, value in label_metrics.items():
                all_metrics[label][metric].append(value)

    # 计算每个类的平均指标
    avg_metrics = {label: {metric: np.mean(values) for metric, values in label_metrics.items()}
                   for label, label_metrics in all_metrics.items()}

    # 将结果保存到文件
    save_metrics_to_file(avg_metrics, output_file)
    print(f"Metrics saved to {output_file}")


def save_metrics_to_file(metrics, output_file):
    """
    Save the metrics dictionary to a text file.
    """
    with open(output_file, 'w') as f:
        for label, label_metrics in metrics.items():
            f.write(f"Label {label}:\n")
            for metric, value in label_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", type=str, default='label', help="Flag for running fine-tuning")
    parser.add_argument("--weight", type=float, default=0.0)
    args = parser.parse_args()
    print(f"Running Fine-Tuning with flag: {args.flag}, {args.weight}")


    # Set the paths to the ground truth and predicted label folders
    FLARE_dir = "/home/yuying/abdomen_seg/FLARE"
    ground_truth_folder = os.path.join(FLARE_dir, "Testing", "labels")
    predicted_folder = "./outputs_tuning/predictions_{}_{}".format(args.flag, args.weight)
    
    # Process the folders and compute metrics
    output_file = './outputs_tuning/metrics_{}_{}.txt'.format(args.flag, args.weight)
    avg_metrics = process_folder(ground_truth_folder, predicted_folder, output_file=output_file)
