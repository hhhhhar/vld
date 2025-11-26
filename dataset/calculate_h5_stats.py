import h5py
import numpy as np
import json
import glob
import os

def calculate_h5_stats(data_dir):
    """
    遍历目录下所有 .h5 文件，计算 poses 的全局 min 和 max
    """
    file_list = glob.glob(os.path.join(data_dir, "*.h5"))
    if not file_list:
        print(f"错误: 在 {data_dir} 下没有找到 .h5 文件")
        return

    all_poses = []
    
    print(f"正在扫描 {len(file_list)} 个文件...")
    for fpath in file_list:
        try:
            with h5py.File(fpath, 'r') as f:
                # 读取 poses: (30, 1, 7) -> reshape to (30, 7)
                poses = f['poses'][:]
                poses = poses.reshape(-1, 7)
                all_poses.append(poses)
        except Exception as e:
            print(f"跳过损坏文件 {fpath}: {e}")

    # 拼接所有数据
    all_poses = np.concatenate(all_poses, axis=0) # (Total_Frames, 7)
    
    stats = {
        "min": all_poses.min(axis=0).tolist(),
        "max": all_poses.max(axis=0).tolist()
    }
    
    print("=== 统计完成 ===")
    print("Action Dim:", all_poses.shape[1])
    print("Min:", stats['min'])
    print("Max:", stats['max'])
    
    # 保存为 json
    with open("action_stats.json", "w") as f:
        json.dump(stats, f)
    print("统计量已保存至 action_stats.json")

# 使用示例
if __name__ == "__main__":
    # 替换为你存放 h5 文件的文件夹路径
    DATA_DIR = "/home/hhhar/liuliu/vld/data/regression/data_res/" 
    calculate_h5_stats(DATA_DIR)