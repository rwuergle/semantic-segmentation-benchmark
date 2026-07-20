# semantic-segmentation-benchmark
This repository presents a method to benchmark different point cloud classifier, especially for *aerial laser scanning* but can be extended to any kind of point clouds. 
## Quick Links
* [Documentation](docs/README.md)
* [Installation](#installation)

## Installation

Clone the repository and install the package locally from the project root.

```bash
git clone https://github.com/rwuergle/semantic-segmentation-benchmark.git
cd semantic-segmentation-benchmark
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If you prefer a normal install instead of editable mode:

```bash
pip install .
```

If you want to install the full development environment from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage
A example notebook can be found [here](./tests/test.ipynb).

```python
from semantic_segmentation_benchmark.metrics import Benchmark
import os
from tqdm import tqdm

ground_truth_dir =r"D:\data\1_reference_tiles"
predictions_dir = r"D:\data\1_predicted_tiles"

for file in tqdm(os.listdir(predictions_dir), unit="file", desc="Generating reports"):
    if (not file.endswith(".laz")) and (not file.endswith(".las")):
        continue
    
    predictions_path = os.path.join(predictions_dir, file)
    # files are named the same in reference and predicted
    ground_truth_path = os.path.join(ground_truth_dir, file)

    remap_ground_truth = {"default": 0, 2: 2, 18: 2, 31: 2, 3: 3, 4: 3, 5: 3, 6: 6, 21: 21, 22: 22, 26: 26}
    remap_predictions = {"default": "keep", 29:0}

    Benchmark.generate_report(predictions_path, ground_truth_path,          remap_ground_truth=remap_ground_truth,  remap_predictions=remap_predictions, model_name="classical", matching_precision=3, use_kdtree_matching = True)

    # output file are created in "./evaluations"
```

## License

This project is licensed under the [MIT License](LICENSE)