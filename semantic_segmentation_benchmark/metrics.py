import numpy as np
from semantic_segmentation_benchmark.constants import PROJECT_CLASSIFIED_MAP
from semantic_segmentation_benchmark.helper import loaded_copc_to_las, write_copc_from_lasdata
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, cohen_kappa_score
import seaborn as sns
import json
import laspy
import os
from scipy.spatial import cKDTree


class Benchmark:
    def __init__(self):
        pass
    
    @staticmethod
    def generate_report(predictions_pc_dir: str, ground_truth_pc_dir: str, model_name: str = "default", tile_name: str = "default",
                        remap_predictions: dict | None = None, remap_ground_truth: dict | None = None,
                        classification_map: dict = PROJECT_CLASSIFIED_MAP, output_folder: str = "./evaluations", matching_precision: float = 3.0, use_kdtree_matching: bool = False) -> None:

        pc_ground_truth = laspy.read(ground_truth_pc_dir)
        pc_predictions = laspy.read(predictions_pc_dir)

        if tile_name == "default":
            tile_name = os.path.basename(ground_truth_pc_dir).split(".")[0]

        all_labels = list(classification_map.keys())
        target_names = list(classification_map.values())

        predictions, ground_truth = Benchmark.get_pred_gt(pc_predictions, pc_ground_truth, remap_predictions=remap_predictions, remap_ground_truth=remap_ground_truth, matching_precision=matching_precision, use_kdtree_matching=use_kdtree_matching)

        balanced_acc = balanced_accuracy_score(ground_truth, predictions)
        kappa = cohen_kappa_score(ground_truth, predictions)
        report = classification_report(ground_truth, predictions, labels=all_labels, target_names=target_names, digits=4, output_dict=True, zero_division=0)

        cm = confusion_matrix(ground_truth, predictions, labels=all_labels)

        metrics = {
            "balanced_accuracy": float(balanced_acc),
            "cohen_kappa": float(kappa),
            "report_dict": report,
            "confusion_matrix": cm.tolist(),
        }

        save_path: str = os.path.join(output_folder, f"{tile_name}_{model_name}.json")
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
    def get_classification_matching_indices(pc_predictions: laspy.LasData, pc_ground_truth: laspy.LasData, precision: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
        coords_pred = np.round(pc_predictions.xyz, int(precision))
        coords_gt = np.round(pc_ground_truth.xyz, int(precision))

        dtype = np.dtype([("x", coords_pred.dtype), ("y", coords_pred.dtype), ("z", coords_pred.dtype)])

        pred_struct = np.ascontiguousarray(coords_pred).view(dtype).ravel()
        gt_struct = np.ascontiguousarray(coords_gt).view(dtype).ravel()

        order = np.argsort(pred_struct)
        change_indices = order[np.searchsorted(pred_struct[order], gt_struct)]

        ground_truth = pc_ground_truth.classification
        predictions = pc_predictions.classification[change_indices]

        return predictions, ground_truth
    

    @staticmethod
    def get_pred_gt(pc_predictions: laspy.LasData, pc_ground_truth: laspy.LasData, remap_predictions: dict | None = None, remap_ground_truth: dict | None = None, matching_precision: float = 3.0, use_kdtree_matching: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if use_kdtree_matching:
            predictions, ground_truth = Benchmark.get_classification_matching_indices_kdtree(pc_predictions, pc_ground_truth, matching_precision)
        else:
            try:
                predictions, ground_truth = Benchmark.get_classification_matching_indices(pc_predictions, pc_ground_truth, precision=matching_precision)
            except:
                print("Error in matching indices. Falling back to KDTree matching.")
                predictions, ground_truth = Benchmark.get_classification_matching_indices_kdtree(pc_predictions, pc_ground_truth, matching_precision)

        if remap_ground_truth is not None:
            ground_truth = Benchmark.remap(ground_truth, remap_ground_truth)
        if remap_predictions is not None:
            predictions = Benchmark.remap(predictions, remap_predictions)
        
        return predictions, ground_truth
    

    @staticmethod
    def get_classification_matching_indices_kdtree(pc_predictions: laspy.LasData, pc_ground_truth: laspy.LasData, matching_precision: float) -> tuple[np.ndarray, np.ndarray]:

        pred_xyz = np.asarray(pc_predictions.xyz)
        gt_xyz = np.asarray(pc_ground_truth.xyz)

        tree = cKDTree(pred_xyz)

        distances, indices = tree.query(gt_xyz, k=1)

        max_distance = 10 ** (-matching_precision)

        if max_distance is not None:
            valid = distances <= max_distance
            predictions = pc_predictions.classification[indices[valid]]
            ground_truth = pc_ground_truth.classification[valid]
        else:
            predictions = pc_predictions.classification[indices]
            ground_truth = pc_ground_truth.classification

        return predictions, ground_truth
    
    @staticmethod
    def generate_visual_comparaison(predictions_pc_dir: str, ground_truth_pc_dir: str, output_path: str | None = None, remap_predictions: dict | None = None, remap_ground_truth: dict | None = None) -> None:
        
        if output_path is None:
            output_path = predictions_pc_dir.split('.')[0] + "_comparaison.copc.laz"

        pc_ground_truth = laspy.read(ground_truth_pc_dir)
        pc_predictions = laspy.read(predictions_pc_dir)

        predictions, ground_truth = Benchmark.get_pred_gt(pc_predictions, pc_ground_truth, remap_predictions=remap_predictions, remap_ground_truth=remap_ground_truth)

        classification = np.where(predictions == ground_truth, 3, 21)

        pc_output = loaded_copc_to_las(pc_ground_truth)
        pc_output.classification = classification
        write_copc_from_lasdata(pc_output, output_path)
    
    def generate_visual_bimodel_comparaison(predictions_pc1_dir: str, predictions_pc2_dir: str, ground_truth_pc_dir: str, output_path: str | None = None, remap1_predictions: dict | None = None, remap2_predictions: dict | None = None, remap_ground_truth: dict | None = None) -> None:
        
        if output_path is None:
            output_path = predictions_pc1_dir.split('.')[0] + "_comparaison.copc.laz"

        pc_ground_truth = laspy.read(ground_truth_pc_dir)
        pc1_predictions = laspy.read(predictions_pc1_dir)
        pc2_predictions = laspy.read(predictions_pc2_dir)

        predictions1, ground_truth = Benchmark.get_pred_gt(pc1_predictions, pc_ground_truth, remap_predictions=remap1_predictions, remap_ground_truth=remap_ground_truth)
        predictions2, ground_truth = Benchmark.get_pred_gt(pc2_predictions, pc_ground_truth, remap_predictions=remap2_predictions, remap_ground_truth=remap_ground_truth)

        conditions = [
        (predictions1 == ground_truth) & (predictions2 == ground_truth),  # All three match
        (predictions1 == ground_truth) & (predictions2 != ground_truth),  # Only nb1 matches GT
        (predictions2 == ground_truth) & (predictions1 != ground_truth),  # Only nb2 matches GT
        (predictions1 == predictions2) & (predictions2 != ground_truth),   # both agree but defer from GT
        (predictions1 != predictions2) & (predictions1 != ground_truth) & (predictions2 != ground_truth)   # Neither matches GT
        ]

        choices = [3, 26, 2, 0, 21]

        classification = np.select(conditions, choices, default=0)


        pc_output = loaded_copc_to_las(pc_ground_truth)
        pc_output.classification = classification
        write_copc_from_lasdata(pc_output, output_path)