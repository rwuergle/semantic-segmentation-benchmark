import laspy
import subprocess
import json
import tempfile
import os
import shutil


def copc_to_laz(input_path: str, output_path: str, do_compress: bool = True):

    pc = laspy.read(input_path)

    if pc.vlrs is not None:
        pc.header.vlrs.extract("CopcInfoVlr")

    if pc.evlrs is not None:
        pc.evlrs.extract("CopcHierarchyVlr")
    
    if do_compress:
        if output_path.endswith(".las"):
            output_path = output_path.replace(".las", ".laz")
    else:
        if output_path.endswith(".laz"):
            output_path = output_path.replace(".laz", ".las")

    pc.write(output_path, do_compress=do_compress)


def loaded_copc_to_las(loaded_copc: laspy.LasData) -> laspy.LasData:
    if loaded_copc.vlrs is not None:
        loaded_copc.header.vlrs.extract("CopcInfoVlr")

    if loaded_copc.evlrs is not None:
        loaded_copc.evlrs.extract("CopcHierarchyVlr")
    
    return loaded_copc


def write_copc_from_lasdata(las_data: laspy.LasData, output_path: str) -> None:
    pdal_exe = shutil.which("pdal")
    if pdal_exe is None:
        raise RuntimeError("PDAL executable not found. Writing copc failed. Install PDAL and add it to PATH.")

    with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        las_data.write(tmp_path)

        pipeline = {
            "pipeline": [
                tmp_path,
                {
                    "type": "writers.copc",
                    "filename": output_path
                }
            ]
        }

        subprocess.run([pdal_exe, "pipeline", "--stdin"], input=json.dumps(pipeline), text=True, check=True)

    finally:
        os.remove(tmp_path)