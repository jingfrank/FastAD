import torch
import torch.nn.functional as F

def compute_cdf_and_binary_features(
    features, n_quantiles=100, min_q=0.01, max_q=0.99,

    quantile_method="channelwise",
):
    """
    计算累积分布函数(CDF)和二值化特征集合

    Args:
        features: [B*N, C, P*P] 的特征张量
        n_quantiles: 分位数的数量
        min_q: 每个通道用于确定阈值范围的下分位点
        max_q: 每个通道用于确定阈值范围的上分位点
        quantile_method: "channelwise" 逐通道降低峰值内存，
            "vectorized" 在输入设备上并行计算所有通道


    Returns:
        cdf: [B*N, C*n_quantiles] 累积分布函数值

    """
    if quantile_method not in ("channelwise", "vectorized"):
        raise ValueError("quantile_method must be 'channelwise' or 'vectorized'")
    device = features.device
    B_N, C, P_P = features.shape
    eps = 1e-6
    # features = features / torch.sqrt(torch.sum(features**2, dim=-1, keepdim=True) + eps)
    input_dtype = features.dtype  # 记录输入数据类型
    if features.dtype == torch.float16:
        features = features.float()

    # 两种路径均使用每个通道的全部样本位置，不分块或采样。
    flattened = features.permute(0, 2, 1).reshape(-1, C)
    quantile_levels = features.new_tensor([min_q, max_q])
    if quantile_method == "vectorized":
        min_vals, max_vals = torch.quantile(flattened, quantile_levels, dim=0)
    else:
        channel_quantiles = []
        for i in range(C):
            # 同时请求上下分位点，避免同一通道重复排序。
            channel_quantiles.append(torch.quantile(flattened[:, i], quantile_levels))
        min_vals, max_vals = torch.stack(channel_quantiles, dim=1)

    # Step 2: 初始化输出张量
    cdf = torch.zeros((B_N, C, n_quantiles),device = device)              # [B*N, C, n_quantiles]

    temperature = 0.01  # 越小越接近硬二值化，越大越平滑
    # Step 3: 对每个分位数计算阈值和二值化
    for q in range(n_quantiles):
        # 计算第q个分位数的阈值
        threshold = min_vals + (max_vals - min_vals) * (q + 1) / (n_quantiles + 1)
        # threshold: [C]
        # threshold = torch.log(1 + torch.exp(threshold))  # 使用softplus确保阈值为正
        # 扩展阈值维度以匹配features的形状

        threshold_expanded = threshold.unsqueeze(0).unsqueeze(2)  # [1, C, 1]

        # 二值化：判断features中每个元素是否小于对应channel的阈值
        binary_mask = (features >= threshold_expanded).float()  # [B*N, C, P*P]

        # 计算CDF：在P*P维度上取均值（即在空间维度上的比例）
        cdf[:, :, q] = binary_mask.mean(dim=2)  # [B*N, C]

    # 将cdf重塑为 [B*N, C*n_quantiles]
    cdf_final = cdf.permute(0, 2, 1).reshape(B_N, -1)  # [B*N, C*n_quantiles]
    # print(f"Output cdf shape: {cdf_final.shape}, binary_set shape: {binary_set_final.shape}")
    return cdf_final
