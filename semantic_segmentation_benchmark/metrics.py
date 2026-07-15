import numpy as np
from semantic_segmentation_benchmark.constants import PROJECT_CLASSIFIED_MAP
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, cohen_kappa_score
import seaborn as sns
import json
import os


class Metrics:
    def __init__(self):
        pass
    
    @staticmethod
    def generate_report(predictions: np.ndarray, ground_truth: np.ndarray, tile_name: str = "default", model_name: str = "default", 
                        remap_predictions: dict | None = None, remap_ground_truth: dict | None = None,
                        classification_map: dict = PROJECT_CLASSIFIED_MAP) -> None:

        all_labels = list(classification_map.keys())
        target_names = list(classification_map.values())

        if remap_ground_truth is not None:
            ground_truth = Metrics.remap(ground_truth, remap_ground_truth)
        if remap_predictions is not None:
            predictions = Metrics.remap(predictions, remap_predictions)

        balanced_acc = balanced_accuracy_score(ground_truth, predictions)
        kappa = cohen_kappa_score(ground_truth, predictions)
        report = classification_report(ground_truth, predictions, labels=all_labels, target_names=target_names, digits=4, output_dict=True)

        cm = confusion_matrix(ground_truth, predictions, labels=all_labels)

        metrics = {
            "balanced_accuracy": float(balanced_acc),
            "cohen_kappa": float(kappa),
            "report_dict": report,
            "confusion_matrix": cm.tolist(),
            "labels": all_labels,
            "class_names": target_names,
        }

        save_path: str = f"./evaluations/{tile_name}_{model_name}.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)
    
    @staticmethod
    def remap(labels: np.ndarray, mapping: dict) -> np.ndarray:
        default = mapping.get("default", None)

        if default == "keep":
            return np.vectorize(lambda x: mapping.get(x, x))(labels)
        elif default is not None:
            return np.vectorize(lambda x: mapping.get(x, default))(labels)
        else:
            return np.vectorize(mapping.get)(labels)

    @staticmethod
    def get_matching_indices(pc_predictions, pc_ground_truth, precision=3) -> tuple[np.ndarray, np.ndarray]:
        pred_lookup = {tuple(p): i for i, p in enumerate(np.round(pc_predictions.xyz,precision))}
        change_indices = np.array([pred_lookup[tuple(np.round(p,precision))] for p in pc_ground_truth.xyz])
        ground_truth = pc_ground_truth.classification
        predictions = pc_predictions[change_indices].classification
        return predictions, ground_truth