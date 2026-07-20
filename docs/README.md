# Documentation
The aim of this library is to compare the metrics of different point cloud classifiers. In order to do so, so called reports are created while comparing the predicted point cloud to the ground truth. those can then be imported, visualised and compared with [tile_classification_dashboard.html](../tile_classification_dashboard.html)

![Tile 2563000_1208500 classification (2022 top and minkunet bottom)](./images/point_cloud_pred_vs_gt.png)


## 1. Benchmark
The class `benchmark` from the module `semantic_segmentation_benchmark.metrics` handles point cloud formats ``(copc).laz`` and ``.las``. 
### 1.1 generate_report()
The `generate_report` static method takes at least 2 input arguments: `predictions_pc_dir`, the path to the .las/.laz file for the predicted classification point cloud and `ground_truth_pc_dir` the point cloud with the ground truth classification. Furthermore, if the point clouds have different class id, they must be remapped, thanks to ``remap_ground_truth`` and ``remap_predictions`` to a common reference which is defined as ``classification_map``. Per default, the classes provided are the following: 

| ID | Class |
| :--- | :----------------------- |
| **0**  | Other                    |
| **2**  | Ground                   |
| **3**  | vegetation               |
| **6**  | Building roofs           |
| **21** | Cars                     |
| **22** | Building facades         |
| **26** | Roof structures          |

Notice that the remapping for the predictions is dependant on the classifier. For example, the flai classifier remapping will look like this

| Predicted ID | Remapped ID |
| :------------ | :------------------- |
| **default**   | 0                    |
| **2**         | 2                    |
| **3**         | 3                    |
| **6**         | 6                    |
| **21**        | 21                   |
| **22**        | 22                   |
| **20**        | 26                   |

When creating thoses maps in the package, they have the keyword *default*, which tells the values not listed in the dictionnary which remapped value they take. There are two options:
* "default" : integer => all values not listed in the dictionnary will take this integer's value
* "default": "keep" => the values not listed in the dictionnary will be kept as is
* is the key "default" is not provided, all values that are not a key will be mapped to 0

its important to be careful while remapping, since some classifier dont predict the same number of classes. For example, in the 7 classes benchmark, *pointly* classifier actually only predicts 5 classes (the building class is aggregated). So multiple ground truth id must be remapped to the same id. This ground truth reclassification map is classed `SITN_POINTLY_REMAP` and is available in `semantic_segmentation_benchmark.constants` like other reclassification map (flai, minkunet SITN, classical SITN, deep learning SITN, pointly as well as the two ground truth remap (pointly remap and other remap)).

Another important point to mention is that most of point clouds have generated a slight $\Delta r$ with r the position (x,y,z). So the points are not exactly alligned, and have to be matched, since their index most likely differs. There are two proposed methods: bucket matching (faster, per default) and kd_tree matching, which is required if $\Delta r$ is significant. Just add the keyword argument `use_kdtree_matching=True`. In any cases, its also possible to add the ``matching_precision`` (max distance or rounding is $10^{-\text{matching\_precision}}$) which is per default 3.0 (float). Notice that in ambiguous cases, some points might be removed.

The method then continues to compute some metrics which will finally be added to the *.json*. the file may then be drag and dropped into [tile_classification_dashboard.html](../tile_classification_dashboard.html), which is the visualisation of the results. 