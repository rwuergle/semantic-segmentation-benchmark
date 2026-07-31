from pathlib import Path
import laspy

def combine_laz_files(input_folder_path, output_file_path, chunk_size=1_000_000):
    """
    Combines multiple small .laz files into a single .laz file.
    
    Parameters:
        input_folder_path (str or Path): Directory containing the small LAZ files.
        output_file_path (str or Path): Filename/path for the combined output.
        chunk_size (int): Number of points to stream at a time (keeps RAM low).
    """
    input_path = Path(input_folder_path)
    output_path = Path(output_file_path)
    
    # 1. Gather all .laz and .las files in the directory
    laz_files = list(input_path.glob("*.laz")) + list(input_path.glob("*.las"))
    
    if not laz_files:
        print(f"❌ Error: No .laz or .las files found in '{input_folder_path}'")
        return

    print(f"📂 Found {len(laz_files)} files to merge.")
    print("⏳ Initializing master header template...")

    # 2. Open the first file to grab the header template
    with laspy.open(laz_files[0], mode="r") as template_reader:
        master_header = template_reader.header

    # 3. Create the new output file and stream points into it
    print(f"💾 Creating output file: {output_path.name}")
    
    with laspy.open(output_path, mode="w", header=master_header) as writer:
        for idx, file_path in enumerate(laz_files, start=1):
            print(f"   [{idx}/{len(laz_files)}] Appending: {file_path.name}...")
            
            with laspy.open(file_path, mode="r") as reader:
                # Check for potential point format mismatches
                if reader.header.point_format != master_header.point_format:
                    print(f"⚠️ Warning: {file_path.name} has a different point format. Skipping file.")
                    continue
                
                # Stream the points in chunks to keep memory usage minimal
                for points_chunk in reader.chunk_iterator(chunk_size):
                    writer.write_points(points_chunk)

    print(f"🎉 Success! All files successfully merged into: {output_path}")

# =====================================================================
# Execution Block
# =====================================================================
if __name__ == "__main__":
    # Change these paths to match your folder setup
    INPUT_DIR = r"D:\Raphael\point-cloud-classifier\visualization\simon"
    OUTPUT_FILE = r"D:\Raphael\point-cloud-classifier\visualization\combined_output_file.laz"
    
    # Optional: Create the input directory layout if testing
    Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Run the merger
    combine_laz_files(INPUT_DIR, OUTPUT_FILE)
