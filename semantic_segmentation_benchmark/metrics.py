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
            return np.vectorize(lambda x: mapping.get(x, 0))(labels)

    @staticmethod
    def get_classification_matching_indices(pc_predictions: laspy.LasData, pc_ground_truth: laspy.LasData, precision: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
        
        scale = 10 ** int(precision)

        pred_xyz = np.asarray(pc_predictions.xyz)
        gt_xyz = np.asarray(pc_ground_truth.xyz)

        pred_i = np.round(pred_xyz * scale).astype(np.int64)
        gt_i = np.round(gt_xyz * scale).astype(np.int64)

        mins = np.minimum(pred_i.min(axis=0), gt_i.min(axis=0))
        pred_i -= mins
        gt_i -= mins

        maxs = np.maximum(pred_i.max(axis=0), gt_i.max(axis=0))
        bits = np.maximum(np.ceil(np.log2(maxs.astype(np.float64) + 1)).astype(np.int64), 1)

        if bits.sum() > 63:
            raise ValueError(
                f"Coordinate range needs {int(bits.sum())} bits at precision={precision}, "
                f"which doesn't fit in a uint64 key. Lower matching_precision or use "
                f"KDTree matching."
            )

        shift_y = int(bits[2])
        shift_x = shift_y + int(bits[1])

        def pack(xyz_i: np.ndarray) -> np.ndarray:
            x = xyz_i[:, 0].view(np.uint64)
            y = xyz_i[:, 1].view(np.uint64)
            z = xyz_i[:, 2].view(np.uint64)
            keys = x << np.uint64(shift_x)
            keys |= y << np.uint64(shift_y)
            keys |= z
            return keys

        pred_keys = pack(pred_i)
        gt_keys = pack(gt_i)
        del pred_i, gt_i

        sorter = np.argsort(pred_keys)
        pred_sorted = pred_keys[sorter]
        del pred_keys

        gt_order = np.argsort(gt_keys)
        gt_sorted = gt_keys[gt_order]

        pos_sorted = np.searchsorted(pred_sorted, gt_sorted)
        np.clip(pos_sorted, 0, len(pred_sorted) - 1, out=pos_sorted)

        matched_sorted = pred_sorted[pos_sorted] == gt_sorted

        pos = np.empty_like(pos_sorted)
        pos[gt_order] = pos_sorted
        valid = np.empty_like(matched_sorted)
        valid[gt_order] = matched_sorted

        matched_pred_idx = sorter[pos[valid]]
        matched_gt_idx = np.nonzero(valid)[0]

        n_unmatched = len(gt_keys) - int(valid.sum())
        if n_unmatched:
            print(f"[get_classification_matching_indices] {n_unmatched}/{len(gt_keys)} "
                f"ground truth points had no exact match in predictions at precision={precision}.")

        predictions = pc_predictions.classification[matched_pred_idx]
        ground_truth = pc_ground_truth.classification[matched_gt_idx]

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
    
class SpatialBenchmark:
    def __init__(self):
        pass

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
    
    @staticmethod
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