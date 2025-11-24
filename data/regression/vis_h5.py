import h5py
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

def print_h5_tree(h5_path):
    def visitor(name, obj):
        # 计算缩进层级
        depth = name.count('/')
        indent = "  " * depth

        if isinstance(obj, h5py.Group):
            print(f"{indent}+ [G] {name}/")
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            print(f"{indent}- [D] {name}    shape={shape} dtype={dtype}")

    with h5py.File(h5_path, 'r') as f:
        print(f"H5 file: {h5_path}")
        f.visititems(visitor)
        plt.figure(figsize=(10, 6))
        for img in f["rgb"]:
            plt.imshow(img)
            plt.show()

print_h5_tree("/home/hhhar/liuliu/vld/data/regression/data_res/data_0001.h5")