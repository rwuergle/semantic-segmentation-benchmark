import numpy as np
import laspy

# Original mapping (Key: New Class, Value: Old Class)
original_remap = {
    2: 1,
    3: 3,
    4: 3,
    5: 3,
    6: 4,
    7: 1,
    9: 2,
    14: 5,
    15: 6,
    17: 7,
    18: 0,
    19: 8,
    21: 9,
    22: 10,
    25: 11,
    26: 12,
    29: 13,
    31: 1,
}

# Invert the map so it is (Old Class -> New Class)
value_to_key_map = {v: k for k, v in original_remap.items()}


def remap_laz_classification(input_path, output_path, mapping):
    # FIXED: laspy.read does not use 'with' context manager
    las = laspy.read(input_path)

    # Extract the classification array
    classifications = np.array(las.classification)

    # Create a vectorized mapping array for fast performance
    max_val = max(max(mapping.keys()), classifications.max())
    lookup_array = np.arange(max_val + 1)

    # Populate the lookup array with the new classification targets
    for old_val, new_val in mapping.items():
        lookup_array[old_val] = new_val

    # Perform the remapping across all points using NumPy indexing
    new_classifications = lookup_array[classifications]

    # Create a new file using the same header format
    new_las = laspy.LasData(las.header)

    # FIXED: Safely copy points and assign the new classification array
    new_las.points = las.points
    new_las.classification = new_classifications.astype(np.uint8)

    # Write the modified data to disk
    new_las.write(output_path)


# Paths to your files
INPUT_LAZ = r"D:\Raphael\point-cloud-classifier\visualization\combined_output_file.laz"
OUTPUT_LAZ = (
    r"D:\Raphael\point-cloud-classifier\visualization\combined_output_file2.laz"
)

remap_laz_classification(INPUT_LAZ, OUTPUT_LAZ, value_to_key_map)
print("Remapping complete successfully!")
