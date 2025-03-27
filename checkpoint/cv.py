# import torch

# # 1️⃣ 加载原始 checkpoint
# checkpoint_path = "/home/user/xfx_map_align/lisa-hdmap/checkpoint/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth"
# checkpoint = torch.load(checkpoint_path, map_location="cpu")

# # 2️⃣ 定义需要调整的键值和目标形状的映射规则
# reshape_rules = {
#     "pts_middle_encoder.conv_input.0.weight": (0, 1, 2, 3, 4),  # [16, 3, 3, 3, 5] -> [3, 3, 3, 5, 16]
#     "pts_middle_encoder.encoder_layers.encoder_layer1.0.conv1.weight": (0, 1, 2, 3, 4),  # [16, 3, 3, 3, 16] -> [3, 3, 3, 16, 16]
#     "pts_middle_encoder.encoder_layers.encoder_layer1.0.conv2.weight": (0, 1, 2, 3, 4),  # 同上
#     "pts_middle_encoder.encoder_layers.encoder_layer1.1.conv1.weight": (0, 1, 2, 3, 4),  # 同上
#     "pts_middle_encoder.encoder_layers.encoder_layer1.1.conv2.weight": (0, 1, 2, 3, 4),  # 同上
#     "pts_middle_encoder.encoder_layers.encoder_layer1.2.0.weight": (0, 1, 2, 3, 4),  # [32, 3, 3, 3, 16] -> [3, 3, 3, 16, 32]
#     "pts_middle_encoder.encoder_layers.encoder_layer2.0.conv1.weight": (0, 1, 2, 3, 4),  # [32, 3, 3, 3, 32] -> [3, 3, 3, 32, 32]
#     "pts_middle_encoder.encoder_layers.encoder_layer2.0.conv2.weight": (0, 1, 2, 3, 4),  # 同上
#     "pts_middle_encoder.encoder_layers.encoder_layer2.1.conv1.weight": (0, 1, 2, 3, 4),  # 同上
#     "pts_middle_encoder.encoder_layers.encoder_layer2.1.conv2.weight": (0, 1, 2, 3, 4),  # 同上
#     "pts_middle_encoder.encoder_layers.encoder_layer2.2.0.weight": (0, 1, 2, 3, 4),  # [64, 3, 3, 3, 32] -> [3, 3, 3, 32, 64]
#     "pts_middle_encoder.encoder_layers.encoder_layer3.0.conv1.weight": (0, 1, 2, 3, 4),  # [64, 3, 3, 3, 64] -> [3, 3, 3, 64, 64]
#     "pts_middle_encoder.encoder_layers.encoder_layer3.0.conv2.weight": (0, 1, 2, 3, 4),  # 同上
#     "pts_middle_encoder.encoder_layers.encoder_layer3.1.conv1.weight": (0, 1, 2, 3, 4),  # 同上
#     "pts_middle_encoder.encoder_layers.encoder_layer3.1.conv2.weight": (0, 1, 2, 3, 4),  # 同上
#     "pts_middle_encoder.encoder_layers.encoder_layer3.2.0.weight": (0, 1, 2, 3, 4),  # [128, 3, 3, 3, 64] -> [3, 3, 3, 64, 128]
#     "pts_middle_encoder.encoder_layers.encoder_layer4.0.conv1.weight": (0, 1, 2, 3, 4),  # [128, 3, 3, 3, 128] -> [3, 3, 3, 128, 128]
#     "pts_middle_encoder.encoder_layers.encoder_layer4.0.conv2.weight": (0, 1, 2, 3, 4),  # 同上
#     "pts_middle_encoder.encoder_layers.encoder_layer4.1.conv1.weight": (0, 1, 2, 3, 4),  # 同上
#     "pts_middle_encoder.encoder_layers.encoder_layer4.1.conv2.weight": (0, 1, 2, 3, 4),  # 同上
#     "pts_middle_encoder.conv_out.0.weight": (0, 1, 2, 3, 4),  # [128, 1, 1, 3, 128] -> [1, 1, 3, 128, 128]
# }

# # 3️⃣ 遍历 state_dict 中的所有参数
# for key in list(checkpoint["state_dict"].keys()):  # 使用 list 避免修改字典时的迭代问题
#     if key in reshape_rules:
#         original_tensor = checkpoint["state_dict"][key]
#         original_shape = original_tensor.shape
#         target_rule = reshape_rules[key]

#         # 检查是否需要调整形状
#         if len(original_shape) == 5:  # 仅处理 5D 张量
#             try:
#                 # 根据规则调整维度顺序
#                 new_tensor = original_tensor.permute(*target_rule)
#                 print(f"Fixed shape for: {key} | Old Shape: {original_shape} | New Shape: {new_tensor.shape}")
#                 checkpoint["state_dict"][key] = new_tensor
#             except Exception as e:
#                 print(f"Failed to reshape {key}: {e}")
#         else:
#             print(f"Skipping {key} as it is not a 5D tensor (shape: {original_shape})")

# # 4️⃣ 保存修正后的 checkpoint
# fixed_checkpoint_path = "/home/xiefuxin/mmdetection3d-main/checkpoint/bevfusionLC_fixed.pth"
# torch.save(checkpoint, fixed_checkpoint_path)
# print(f"✅ 转换完成，新的权重文件保存在 {fixed_checkpoint_path}")

import torch

# 1️⃣ 加载原始 checkpoint
checkpoint_path = "/home/user/xfx_map_align/lisa-hdmap/checkpoint/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# 2️⃣ 遍历 state_dict 中的所有参数
for key in checkpoint["state_dict"]:
    tensor = checkpoint["state_dict"][key]

    # 只处理 5D 的 Conv3D 权重，跳过 2D Conv 和其他层
    if tensor.dim() == 5:
        print(f"Fixing shape for: {key} | Old Shape: {tensor.shape}")
        
        # 交换维度: [C_out, 3, 3, 3, C_in] -> [3, 3, 3, C_in, C_out]
        tensor = tensor.permute(1, 2, 3, 4, 0)

        # 更新 checkpoint
        checkpoint["state_dict"][key] = tensor
        print(f"New Shape: {tensor.shape}")

# 3️⃣ 保存修正后的 checkpoint
fixed_checkpoint_path = "checkpoint/bevfusionL_fixed.pth"
torch.save(checkpoint, fixed_checkpoint_path)
print(f"✅ 转换完成，新的权重文件保存在 {fixed_checkpoint_path}")
