import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from poutyne import Model
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix


def bootstrap_evaluation_poutyne(model, data, save_logs, n_bootstraps, seed, 
                                 tb_class_index, results_dir=None):
    """
    Perform bootstrap evaluation of a model on a test dataset.

    Args:
        model: The trained Poutyne model to evaluate.
        data: The dataset to evaluate on (e.g., ImageFolder dataset).
        save_logs: Whether to save the metric distributions to CSV.
        n_bootstraps: Number of bootstrap samples to generate.
        seed: Random seed for reproducibility.
        results_dir: Directory to save the bootstrap distribution CSV.

    Returns:
        A pandas DataFrame with mean and confidence intervals for:
        - Accuracy
        - F1 Score
        - Sensitivity (Recall)
        - Specificity
        - Test Loss
    """
    rng = np.random.RandomState(seed)

    # Store bootstrapped metrics
    metrics = {
        "accuracy": [],
        "f1_score": [],
        "sensitivity": [],
        "specificity": [],
        "loss": [],
    }

    # Calculate 10% of the dataset size
    subset_size = int(0.1 * len(data))

    for i in range(n_bootstraps):

        print(f"step {i + 1}/{n_bootstraps}")
        # Sample 10% of the data with replacement
        sampled_indices = rng.choice(len(data), subset_size, replace=True)
        sampled_subset = Subset(data, sampled_indices)
        sampled_loader = DataLoader(sampled_subset, batch_size=32 * 2, shuffle=False)

        # Evaluate the model on the sampled subset
        sample_test_loss, sample_test_acc, sample_y_pred, sample_y_true = model.evaluate_generator(
            sampled_loader, 
            return_pred=True,
            return_ground_truth=True
        )

        sample_sens, sample_spec = tb_metrics_generator(y_pred=sample_y_pred, y_true=sample_y_true, tb_class_index=tb_class_index)

        sample_f1_score = 2 * (sample_sens * sample_spec) / (sample_spec + sample_sens)

        # Append metrics to the list
        metrics["accuracy"].append(sample_test_acc)

        metrics["loss"].append(sample_test_loss)

        metrics["sensitivity"].append(sample_sens)

        metrics["specificity"].append(sample_spec)

        metrics["f1_score"].append(sample_f1_score)



    # Convert metrics to a DataFrame
    metrics_df = pd.DataFrame(metrics)

    if save_logs:
        metrics_df.to_csv(f"{results_dir}/bootstrap_distribution.csv", index=False)


    # Calculate mean and confidence intervals
    mean_metrics = metrics_df.mean()
    confidence_intervals = metrics_df.apply(lambda x: np.percentile(x, [2.5, 97.5]))

    # Create a new DataFrame for mean and confidence intervals
    results_df = pd.DataFrame({
        "metric": mean_metrics.index,
        "mean": mean_metrics.values,
        "lower_ci": confidence_intervals.apply(lambda x: x[0]),  # 2.5th percentile
        "upper_ci": confidence_intervals.apply(lambda x: x[1]),  # 97.5th percentile
    })

    if save_logs:
        metrics_df.to_csv(f"{results_dir}/metrics_df.csv", index=False)


    return results_df


def tb_metrics_generator(y_pred, y_true, tb_class_index):
    """
    Generate sensitivity and specificity for the TB class, given the predicted and true labels.

    Args:
        y_pred: The predicted labels (model output, probability values or logits).
        y_true: The true labels.
        tb_class_index: The index of the TB class (default is 1 for TB).

    Returns:
        sensitivity: Sensitivity for the TB class.
        specificity: Specificity for the TB class.
    """
    
    # Convert predictions to class labels (assuming y_pred is the raw output, e.g., logits or probabilities)
    y_pred_to_class = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_to_class)

    # Define indices for the classes
    if tb_class_index == 0:
        # TB is the first class (index 0)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    else:
        # TB is the second class (index 1)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # Calculate TB sensitivity (Recall)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Avoid division by zero

    # Calculate TB specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero

    return sensitivity, specificity