import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 从同目录下的 features.py 导入核心功能
from features import extract_window_features, aggregate_subject_features

def find_eeg_files(root: Path):
    """递归搜索所有支持的脑电文件格式 (.set, .fif, .edf)"""
    files = []
    for ext in [".set", ".fif", ".edf"]:
        files.extend(root.rglob(f"*{ext}"))
    return files

def load_data_and_extract_features(data_root, window_sec=2.0, step_sec=1.0):
    """
    核心数据流水线：
    1. 读取 participants.tsv 获取分类标签 (Group)
    2. 遍历每个被试的衍生数据 (derivatives)
    3. 裁剪数据并分窗，提取 19 通道的特征
    """
    data_root = Path(data_root)
    participants_tsv = data_root / "participants.tsv"
    
    # 1. 加载被试标签
    df_labels = pd.read_csv(participants_tsv, sep="\t")
    label_map = dict(zip(df_labels["participant_id"], df_labels["Group"]))
    
    # 2. 确定搜索路径 (使用预处理过的衍生数据)
    search_path = data_root / "derivatives"
    files = find_eeg_files(search_path)
    
    X, y = [], []
    
    print(f"开始提取特征... 目标文件数: {len(files)}")
    
    for f_path in files:
        # 从路径中解析被试 ID (例如 sub-001)
        sub_id = None
        for part in f_path.parts:
            if part.startswith("sub-"):
                sub_id = part
                break
        
        if not sub_id or sub_id not in label_map:
            continue
            
        try:
            # 读取脑电数据并裁剪到前 30 秒以加快处理速度
            raw = mne.io.read_raw_eeglab(f_path, preload=True, verbose=False)
            if raw.times[-1] > 30:
                raw.crop(0, 30)
            raw.pick("eeg") # 只要 19 个脑电通道
            
            sfreq = raw.info["sfreq"]
            data = raw.get_data()
            
            # 准备分窗参数
            win_len = int(window_sec * sfreq)
            step_len = int(step_sec * sfreq)
            
            window_feats = []
            # 在信号上滑动窗口
            for start in range(0, data.shape[1] - win_len, step_len):
                window = data[:, start:start + win_len]
                # 提取该窗口的频带功率、熵等特征 (19通道)
                feats, _ = extract_window_features(window, sfreq)
                window_feats.append(feats)
            
            if not window_feats:
                continue
                
            # 将该被试所有窗口的特征聚合为一条向量
            subj_feat = aggregate_subject_features(window_feats)
            
            X.append(subj_feat)
            y.append(label_map[sub_id])
            print(f"  [成功] 被试 {sub_id} 特征提取完成")
            
        except Exception as e:
            print(f"  [跳过] 被试 {sub_id} 处理出错: {e}")

    return np.array(X), np.array(y)

def run_svm_workflow(data_root):
    """运行 SVM 训练与评估的完整工作流"""
    # 1. 准备数据集 (特征 X, 标签 y)
    X, y = load_data_and_extract_features(data_root)
    
    if len(X) == 0:
        print("未找到有效数据，请检查路径。")
        return

    # 2. 定义 SVM 模型流水线 (Pipeline)
    # StandardScaler: 必须步骤！将特征归一化，否则 SVM 无法正确收敛
    # SVC: 支持向量机，使用 RBF (高斯) 核处理非线性分类
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=2.0, probability=True, class_weight="balanced"))
    ])

    # 3. 5折交叉验证 (StratifiedKFold)
    # 确保训练集和测试集的类别比例一致
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_preds = []
    all_trues = []
    
    print("\n--- 开始 5 折交叉验证训练 ---")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练模型
        clf.fit(X_train, y_train)
        # 在测试集上进行预测
        preds = clf.predict(X_test)
        
        all_preds.extend(preds)
        all_trues.extend(y_test)
        
        acc = accuracy_score(y_test, preds)
        print(f"Fold {fold+1} 准确率: {acc:.4f}")

    # 4. 打印最终性能报告
    print("\n=== SVM 最终分类性能报告 ===")
    print(classification_report(all_trues, all_preds))
    print("混淆矩阵 (行代表真实类别，列代表预测类别):")
    print(confusion_matrix(all_trues, all_preds))

if __name__ == "__main__":
    # 自动定位数据路径 (假设数据在项目的 ds004504 目录下)
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "ds004504"
    run_svm_workflow(DATA_PATH)
