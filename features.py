import math
import numpy as np
from scipy.signal import welch, coherence

# 默认的脑电频带范围定义
DEFAULT_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

def permutation_entropy(time_series, m=3, delay=1):
    """
    计算排列熵：衡量时间序列的复杂度和不可预测性
    参数:
        time_series: 单通道时间序列
        m: 模式长度
        delay: 延迟时间
    """
    n = len(time_series)
    if n < m * delay:
        return 0.0
    
    # 针对 m=3 的快速向量化实现
    if m == 3 and delay == 1:
        x = time_series
        s0 = x[:-2]
        s1 = x[1:-1]
        s2 = x[2:]
        S = np.vstack((s0, s1, s2)).T
        perms = np.argsort(S, axis=1)
        # 将排列模式转化为唯一的整数进行计数
        patterns = perms[:, 0] * 100 + perms[:, 1] * 10 + perms[:, 2]
        
        _, counts = np.unique(patterns, return_counts=True)
        probs = counts / counts.sum()
        pe = -np.sum(probs * np.log2(probs + 1e-12))
        return pe / np.log2(math.factorial(m)) # 归一化到 [0, 1]
        
    return 0.0

def hjorth_params(x: np.ndarray):
    """
    计算霍斯参数 (Hjorth Parameters)
    返回: 活动度(Activity), 机动度(Mobility), 复杂度(Complexity)
    """
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    
    activity = var_x
    mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0.0
    complexity = np.sqrt(var_ddx / var_dx) / mobility if var_dx > 0 and mobility > 0 else 0.0
    return activity, mobility, complexity

def spectral_entropy(psd, freqs):
    """
    计算谱熵：衡量功率谱密度的分布均匀程度
    """
    psd = psd.copy()
    psd = psd / np.sum(psd) # 归一化功率谱
    psd = np.clip(psd, 1e-12, None)
    ent = -np.sum(psd * np.log2(psd))
    ent /= np.log2(len(psd)) # 归一化
    return ent

def bandpower_features(data, sfreq, bands=DEFAULT_BANDS):
    """
    计算频带功率特征：包括 5 个频带的绝对功率和相对功率
    参数:
        data: (通道数, 采样点数)
    """
    # 使用 Welch 方法计算功率谱密度 (PSD)
    freqs, psd = welch(data, sfreq, nperseg=min(1024, data.shape[1]))
    total_power = np.trapezoid(psd, freqs, axis=1)
    feats = []
    names = []
    for name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs < fmax)
        # 计算指定频带内的功率（曲线下面积）
        bp = np.trapezoid(psd[:, idx], freqs[idx], axis=1) # 绝对功率
        rbp = bp / np.maximum(total_power, 1e-12)         # 相对功率
        feats.append(bp)
        feats.append(rbp)
        names.append(f"bp_{name}")
        names.append(f"rbp_{name}")
    feats = np.stack(feats, axis=1)  # 形状: (n_channels, n_features)
    return feats, names

def extract_window_features(window, sfreq, bands=DEFAULT_BANDS):
    """
    从单个时间窗口提取所有组合特征 (19通道 x 15特征)
    """
    # 1. 频带功率 (10个特征: 5绝对 + 5相对)
    bp_feats, bp_names = bandpower_features(window, sfreq, bands)

    # 2. 霍斯参数 (3个特征: 活动度、机动度、复杂度)
    hjorth = []
    for ch in range(window.shape[0]):
        activity, mobility, complexity = hjorth_params(window[ch])
        hjorth.append([activity, mobility, complexity])
    hjorth = np.asarray(hjorth)
    hj_names = ["hj_activity", "hj_mobility", "hj_complexity"]

    # 3. 谱熵 (1个特征)
    freqs, psd = welch(window, sfreq, nperseg=min(256, window.shape[1]))
    sent = np.array([spectral_entropy(psd[ch], freqs) for ch in range(window.shape[0])])
    sent = sent[:, None]
    sent_names = ["spec_entropy"]

    # 4. 排列熵 (1个特征)
    pe = []
    for ch in range(window.shape[0]):
        try:
            val = permutation_entropy(window[ch], m=3, delay=1)
        except:
            val = 0.0
        pe.append(val)
    pe = np.array(pe)[:, None]
    pe_names = ["perm_entropy"]

    # 最终合并所有特征，得到 (19, 15) 的特征矩阵
    feats = np.concatenate([bp_feats, hjorth, sent, pe], axis=1)
    names = bp_names + hj_names + sent_names + pe_names
    return feats, names

def aggregate_subject_features(window_features):
    """
    将一个被试所有分窗的特征聚合为最终的特征向量 (均值 + 标准差)
    """
    wf = np.stack(window_features, axis=0)  # 形状: (分窗数, 通道数, 特征数)
    
    # 首先跨分窗聚合（得到该被试在 19 个通道上的平均表现和稳定性）
    mean_w = wf.mean(axis=0) # (19, 15)
    std_w = wf.std(axis=0)   # (19, 15)
    
    # 接着跨通道聚合（将全脑信息压缩为 15 个统计量）
    mean_c = mean_w.mean(axis=0) # (15,)
    std_c = std_w.mean(axis=0)   # (15,)
    
    # 最终特征向量：[均值部分, 标准差部分]，共 30 维 (如果只算15个特征的平均和方差)
    # 注意：在 SVM pipeline 中，19通道的信息会通过均值聚合
    subj_feat = np.concatenate([mean_c, std_c], axis=0)
    return subj_feat
