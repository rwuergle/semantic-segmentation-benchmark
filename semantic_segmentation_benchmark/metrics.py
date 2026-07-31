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
    def generate_visual_bimodel_comparaison(predictions_pc1_dir: str, predictions_pc2_dir: str, ground_truth_pc_dir: str, output_path: str | None = None, remap1_predictions: dict | None = None, remap2_predictions: dict | None = None, remap_ground_truth: dict | None = None,  matching_precision: float = 3.0) -> None:
        
        if output_path is None:
            output_path = predictions_pc1_dir.split('.')[0] + "_comparaison.copc.laz"

        pc_ground_truth = laspy.read(ground_truth_pc_dir)
        pc1_predictions = laspy.read(predictions_pc1_dir)
        pc2_predictions = laspy.read(predictions_pc2_dir)

        predictions1, ground_truth = Benchmark.get_pred_gt(pc1_predictions, pc_ground_truth, remap_predictions=remap1_predictions, remap_ground_truth=remap_ground_truth, matching_precision=matching_precision)
        predictions2, ground_truth = Benchmark.get_pred_gt(pc2_predictions, pc_ground_truth, remap_predictions=remap2_predictions, remap_ground_truth=remap_ground_truth, matching_precision=matching_precision)

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

class ClassificationCombination:
    def __init__(self):
        pass

    @staticmethod
    def match_to_reference(pc_source: laspy.LasData, pc_reference: laspy.LasData, precision: float = 3.0,
                            use_kdtree_matching: bool = False) -> tuple[np.ndarray, np.ndarray]:
        
        ref_xyz = np.asarray(pc_reference.xyz)
        src_xyz = np.asarray(pc_source.xyz)

        if use_kdtree_matching:
            tree = cKDTree(ref_xyz)
            max_distance = 10 ** (-precision)
            distances, ref_idx = tree.query(src_xyz, k=1, distance_upper_bound=max_distance)
            valid = np.isfinite(distances)
            src_idx = np.nonzero(valid)[0]
            ref_idx = ref_idx[valid]
            distances = distances[valid]
        else:
            scale = 10 ** int(precision)

            ref_i = np.round(ref_xyz * scale).astype(np.int64)
            src_i = np.round(src_xyz * scale).astype(np.int64)

            mins = np.minimum(ref_i.min(axis=0), src_i.min(axis=0))
            ref_i = ref_i - mins
            src_i = src_i - mins

            maxs = np.maximum(ref_i.max(axis=0), src_i.max(axis=0))
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

            ref_keys = pack(ref_i)
            src_keys = pack(src_i)
            del ref_i, src_i

            sorter = np.argsort(ref_keys)
            ref_sorted = ref_keys[sorter]

            pos = np.searchsorted(ref_sorted, src_keys)
            np.clip(pos, 0, len(ref_sorted) - 1, out=pos)
            matched = ref_sorted[pos] == src_keys

            src_idx = np.nonzero(matched)[0]
            ref_idx = sorter[pos[matched]]
            distances = np.zeros(len(src_idx))

        if len(ref_idx) == 0:
            return ref_idx, src_idx

        order = np.argsort(distances, kind="stable")
        ref_idx_sorted = ref_idx[order]
        src_idx_sorted = src_idx[order]

        _, first_pos = np.unique(ref_idx_sorted, return_index=True)

        return ref_idx_sorted[first_pos], src_idx_sorted[first_pos]

    @staticmethod
    def combine_majority_vote(pc_dirs: list[str], remaps: list[dict | None] | None = None,
                               reference_pc_dir: str | None = None, output_path: str | None = None,
                               matching_precision: float = 3.0, use_kdtree_matching: bool = False,
                               fallback_to_kdtree: bool = True, unmatched_label: int = 0) -> None:
        """
        Combine the classification of N point clouds (e.g. the outputs of N
        different classifiers run on the same scene) into a single point cloud,
        assigning each point the class that got the most votes.

        The clouds are not assumed to share indices or exact coordinates: points
        may have been slightly displaced by each processing pipeline, and some
        points may be duplicated. Points are therefore matched onto a reference
        point cloud (see `match_to_reference`), and each input cloud contributes
        at most one vote per reference point.

        Parameters
        ----------
        pc_dirs:
            Paths to the N (.copc).laz files to combine.
        remaps:
            Optional list of per-file remap dicts (same length/order as
            `pc_dirs`; use None for a file that needs no remapping). Applied to
            each file's classification via Benchmark.remap before voting -- use
            this to align different classifiers' label schemes onto one common
            scheme.
        reference_pc_dir:
            Path whose point geometry (xyz, and all other output attributes) is
            used for the output cloud. Defaults to pc_dirs[0]. It does not need
            to be one of pc_dirs.
        output_path:
            Where to write the combined point cloud. Defaults to
            "<reference_basename>_majority_vote.copc.laz" next to the reference
            file.
        matching_precision:
            Number of decimal digits used for the grid/bucket match, or the
            KDTree distance threshold (10 ** -matching_precision), when matching
            each cloud onto the reference.
        use_kdtree_matching:
            Use KDTree nearest-neighbour matching for every file instead of the
            faster grid/bucket match. Use this if points were displaced by more
            than a rounding error at `matching_precision`.
        fallback_to_kdtree:
            If the grid/bucket match fails for a file (e.g. the coordinate range
            doesn't fit the packed key at this precision), fall back to KDTree
            matching for that file instead of raising.
        unmatched_label:
            Classification value assigned to reference points that received no
            vote at all (only possible if `reference_pc_dir` is not part of
            `pc_dirs`, or a point falls outside every other cloud's coverage).
        """
        if len(pc_dirs) < 2:
            raise ValueError("combine_majority_vote needs at least 2 point clouds to combine.")

        if remaps is None:
            remaps = [None] * len(pc_dirs)
        if len(remaps) != len(pc_dirs):
            raise ValueError("remaps must have the same length as pc_dirs.")

        if reference_pc_dir is None:
            reference_pc_dir = pc_dirs[0]

        if output_path is None:
            base = os.path.basename(reference_pc_dir).split(".")[0]
            output_path = os.path.join(os.path.dirname(reference_pc_dir) or ".", f"{base}_majority_vote.copc.laz")

        pc_reference = laspy.read(reference_pc_dir)
        n_points = len(pc_reference.classification)

        ref_idx_chunks: list[np.ndarray] = []
        class_chunks: list[np.ndarray] = []

        for pc_dir, remap in zip(pc_dirs, remaps):
            if os.path.abspath(pc_dir) == os.path.abspath(reference_pc_dir):
                pc_source = pc_reference
                ref_idx = np.arange(n_points)
                src_idx = np.arange(n_points)
            else:
                pc_source = laspy.read(pc_dir)
                try:
                    ref_idx, src_idx = ClassificationCombination.match_to_reference(
                        pc_source, pc_reference, precision=matching_precision,
                        use_kdtree_matching=use_kdtree_matching,
                    )
                except ValueError:
                    if not fallback_to_kdtree or use_kdtree_matching:
                        raise
                    print(f"[combine_majority_vote] Bucket matching failed for {pc_dir}, "
                          f"falling back to KDTree matching.")
                    ref_idx, src_idx = ClassificationCombination.match_to_reference(
                        pc_source, pc_reference, precision=matching_precision,
                        use_kdtree_matching=True,
                    )

            if len(ref_idx) < n_points:
                print(f"[combine_majority_vote] {pc_dir}: matched {len(ref_idx)}/{n_points} "
                      f"reference points ({len(src_idx)}/{len(pc_source.classification)} of its "
                      f"own points were used, the rest were unmatched or discarded duplicates).")

            classification = np.asarray(pc_source.classification)[src_idx]
            if remap is not None:
                classification = Benchmark.remap(classification, remap)
            classification = np.asarray(classification, dtype=np.int64)

            ref_idx_chunks.append(ref_idx)
            class_chunks.append(classification)

        all_ref_idx = np.concatenate(ref_idx_chunks)
        all_classes = np.concatenate(class_chunks)

        majority = np.full(n_points, unmatched_label, dtype=pc_reference.classification.dtype)

        if len(all_ref_idx):

            max_class = int(all_classes.max()) + 1
            combined_key = all_ref_idx.astype(np.int64) * max_class + all_classes

            unique_keys, counts = np.unique(combined_key, return_counts=True)
            key_ref_idx = unique_keys // max_class
            key_class = unique_keys % max_class

            order = np.lexsort((key_class, -counts, key_ref_idx))
            key_ref_idx_sorted = key_ref_idx[order]
            key_class_sorted = key_class[order]
            counts_sorted = counts[order]

            _, first_pos = np.unique(key_ref_idx_sorted, return_index=True)
            winning_ref_idx = key_ref_idx_sorted[first_pos]
            winning_class = key_class_sorted[first_pos]
            winning_count = counts_sorted[first_pos]

            majority[winning_ref_idx] = winning_class

            total_votes = np.zeros(n_points, dtype=np.int64)
            np.add.at(total_votes, all_ref_idx, 1)

            n_no_vote = int((total_votes == 0).sum())
            n_no_strict_majority = int(np.sum(winning_count * 2 <= total_votes[winning_ref_idx]))
            if n_no_vote:
                print(f"[combine_majority_vote] {n_no_vote}/{n_points} reference points received "
                      f"no vote from any input cloud and were set to unmatched_label={unmatched_label}.")
            if n_no_strict_majority:
                print(f"[combine_majority_vote] {n_no_strict_majority}/{n_points} reference points "
                      f"had no strict majority (tie broken by lowest class id).")
        else:
            print("[combine_majority_vote] No votes were collected at all; every point was set to "
                  f"unmatched_label={unmatched_label}.")

        pc_output = loaded_copc_to_las(pc_reference)
        pc_output.classification = majority
        write_copc_from_lasdata(pc_output, output_path)