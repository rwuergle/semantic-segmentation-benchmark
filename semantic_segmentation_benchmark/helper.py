import laspy

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