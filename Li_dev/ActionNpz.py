import numpy as np

# 指定你的文件路径
file_path = "/mnt/disk2/ruizhe/Projects/openvlaData/LiData/saved_action_hidden/samples_001000_to_002000.npz"

# 加载 .npz 文件
data = np.load(file_path)

# 遍历所有数组，打印它们的名字、shape 和 dtype
for name in data.files:
    arr = data[name]
    print(f"Array name: {name}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print("-" * 40)
