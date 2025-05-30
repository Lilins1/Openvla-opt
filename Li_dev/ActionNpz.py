import numpy as np
import torch

# 指定你的文件路径
file_path = "/mnt/disk2/ruizhe/Projects/openvlaData/LiData/saved_action_hidden/samples_000630_to_000672.npz"

# 加载 .npz 文件
data = np.load(file_path)

# 遍历所有数组，打印它们的名字、shape 和 dtype
for name in data.files:
    arr = data[name]
    print(f"Array name: {name}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print("-" * 40)

# 宏定义：设置要查看的样本和时间步数
NUM_SAMPLES_TO_SHOW = 100  # 要显示的样本数量
NUM_TIMESTEPS_TO_SHOW = 5  # 每个样本要显示的时间步数

# 打印 mlp_preds 的详细信息
print("\n" + "="*80)
print("mlp_preds 详细信息:")
mlp_preds = data['mlp_preds']
print(f"mlp_preds 形状: {mlp_preds.shape} (样本数, 时间步长, 动作维度)")
print(f"动作维度: {mlp_preds.shape[2]} (可能包含位置、旋转、夹爪状态等)")

# 打印前N个样本的前M个时间步
print(f"\n前{NUM_SAMPLES_TO_SHOW}个样本的前{NUM_TIMESTEPS_TO_SHOW}个时间步的预测动作:")
for i in range(min(NUM_SAMPLES_TO_SHOW, mlp_preds.shape[0])):
    print(f"\n样本 {i+1}:")
    for t in range(min(NUM_TIMESTEPS_TO_SHOW, mlp_preds.shape[1])):
        print(f"mlp_preds  时间步 {t+1}: {mlp_preds[i, t]}")

# 打印 ground_truth 的详细信息
print("\n" + "="*80)
print("ground_truth 详细信息:")
ground_truth = data['ground_truth']
print(f"ground_truth 形状: {ground_truth.shape} (样本数, 时间步长, 动作维度)")

# 打印前N个样本的前M个时间步
print(f"\n前{NUM_SAMPLES_TO_SHOW}个样本的前{NUM_TIMESTEPS_TO_SHOW}个时间步的真实动作:")
for i in range(min(NUM_SAMPLES_TO_SHOW, ground_truth.shape[0])):
    print(f"\n样本 {i+1}:")
    for t in range(min(NUM_TIMESTEPS_TO_SHOW, ground_truth.shape[1])):
        print(f"ground_truth  时间步 {t+1}: {ground_truth[i, t]}")

# 打印 language_instructions 的详细信息
print("\n" + "="*80)
print("language_instructions 详细信息:")
language_instructions = data['language_instructions']
print(f"语言指令数量: {len(language_instructions)}")
print(f"语言指令内容: {language_instructions[0]}")

# 详细对比 mlp_preds 和 ground_truth
print("\n" + "="*80)
print("mlp_preds 与 ground_truth 详细对比:")
print(f"对比前 {NUM_SAMPLES_TO_SHOW} 个样本的前 {NUM_TIMESTEPS_TO_SHOW} 个时间步:")

# 动作维度标签（根据OpenVLA的常见设置）
ACTION_DIM_LABELS = [
    "X位置", "Y位置", "Z位置", 
    "X旋转", "Y旋转", "Z旋转", 
    "夹爪"
]

loss_valueAcc = 0
for i in range(min(NUM_SAMPLES_TO_SHOW, mlp_preds.shape[0])):
    print(f"\n样本 {i+1} 对比:")
    
    # 计算整体差异统计
    diff = mlp_preds[i] - ground_truth[i]  # 形状 (5, 7)
    abs_diff = np.abs(diff)
    mean_abs_diff = np.mean(abs_diff)
    max_abs_diff = np.max(abs_diff)
    
    print(f"  整体平均绝对误差: {mean_abs_diff:.6f}")
    print(f"  整体最大绝对误差: {max_abs_diff:.6f}")

    l1_loss = torch.nn.L1Loss()
    # 确保输入是张量（不是 NumPy 数组）
    ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32)
    mlp_preds_tensor = torch.tensor(mlp_preds, dtype=torch.float32)

    # 计算损失
    loss_value = l1_loss(mlp_preds_tensor, ground_truth_tensor)
    loss_valueAcc += loss_value
    print(f"L1 损失值: {loss_value.item()}")
    
    # 对比每个时间步
    for t in range(min(NUM_TIMESTEPS_TO_SHOW, mlp_preds.shape[1])):
        print(f"\n  时间步 {t+1} 详细对比:")
        print(f"    {'动作维度':<10} {'预测值':<9} {'真实值':<9} {'差异':<9} {'绝对误差':<9}")
        print("-" * 60)
        
        # 对比每个动作维度
        for d in range(mlp_preds.shape[2]):
            pred_val = mlp_preds[i, t, d]
            true_val = ground_truth[i, t, d]
            diff_val = pred_val - true_val
            abs_diff_val = abs(diff_val)
            
            # 使用颜色标记较大的误差
            error_marker = "√"
            # if abs_diff_val > 0.5:
            #     error_marker = "⚠️"
            # elif abs_diff_val < 0.05:
            #     error_marker = "√"
            if pred_val * true_val < 0 and (abs(true_val)+abs(pred_val)) > 0.2:
                error_marker = "⚠️"
            elif abs_diff_val > 0.7:
                error_marker = "!"
            
            print(f"    {ACTION_DIM_LABELS[d]:<12} {pred_val:<12.6f} {true_val:<12.6f} {diff_val:<12.6f} {abs_diff_val:<12.6f} {error_marker}")
        
        # 计算当前时间步的统计 - 修复索引错误
        timestep_diff = diff[t]  # 使用 t 而不是 i,t
        timestep_abs_diff = np.abs(timestep_diff)
        timestep_mean_abs_diff = np.mean(timestep_abs_diff)
        timestep_max_abs_diff = np.max(timestep_abs_diff)
        
        print("-" * 60)
        print(f"    时间步 {t+1} 平均绝对误差: {timestep_mean_abs_diff:.6f}")
        print(f"    时间步 {t+1} 最大绝对误差: {timestep_max_abs_diff:.6f}")

loss_valueAvg = loss_valueAcc / min(NUM_SAMPLES_TO_SHOW, mlp_preds.shape[0])

# 打印所有样本的整体统计信息
print("\n" + "="*80)
print("所有样本的整体统计信息:")

# 计算整体差异
all_diff = mlp_preds - ground_truth
all_abs_diff = np.abs(all_diff)

# 计算整体统计
overall_mean_abs_diff = np.mean(all_abs_diff)
overall_max_abs_diff = np.max(all_abs_diff)

# 计算每个维度的平均绝对误差
dim_mean_abs_diff = np.mean(all_abs_diff, axis=(0, 1))
dim_max_abs_diff = np.max(all_abs_diff, axis=(0, 1))

print(f"整体平均绝对误差: {overall_mean_abs_diff:.6f}")
print(f"整体最大绝对误差: {overall_max_abs_diff:.6f}")
print(f"loss_valueAvg: {loss_valueAvg:.6f}")



print("\n各维度平均绝对误差:")
for d, label in enumerate(ACTION_DIM_LABELS):
    print(f"  {label}: {dim_mean_abs_diff[d]:.6f} (最大误差: {dim_max_abs_diff[d]:.6f})")