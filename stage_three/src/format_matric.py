import csv


def parse_metrics_txt(file_path):
    """
    Parse the metrics.txt file into a dictionary.
    The keys will be label numbers (0-13), and the values will be dictionaries of metrics (DSC, IoU, Precision, Recall).
    """
    metrics = {i: {'DSC': None, 'IoU': None, 'Precision': None, 'Recall': None} for i in range(14)}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_label = None
    for line in lines:
        line = line.strip()
        if line.startswith('Label'):
            # Extract the label number
            current_label = int(line.split()[1].replace(':', ''))
        elif line:
            # Extract metric values (DSC, IoU, Precision, Recall)
            parts = line.split(':')
            metric_name = parts[0].strip()
            metric_value = float(parts[1].strip())
            if metric_name in metrics[current_label]:
                metrics[current_label][metric_name] = metric_value

    return metrics


def save_metrics_to_csv(metrics, output_csv):
    """
    Save the parsed metrics into a CSV file.
    The first column is the metric name, and subsequent columns represent each label.
    """
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Write header row: ["Metric", "Label 0", "Label 1", ..., "Label 13"]
        header = ["Metric"] + [f"Label {i}" for i in range(14)]
        writer.writerow(header)

        # Write rows for each metric (DSC, IoU, Precision, Recall)
        for metric_name in ['DSC', 'IoU', 'Precision', 'Recall']:
            row = [metric_name] + [metrics[label][metric_name] for label in range(14)]
            writer.writerow(row)


# file path
input_file = 'metrics.txt'
output_csv = 'metrics.csv'

metrics = parse_metrics_txt(input_file)
save_metrics_to_csv(metrics, output_csv)

print(f"Metrics saved to {output_csv}")
