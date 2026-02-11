import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# 导入自定义特征模块
from features import extract_window_features, aggregate_subject_features

# 支持的脑电文件扩展名
SUPPORTED_EXTS = [".set", ".fif", ".edf", ".bdf"]

def find_eeg_files(root: Path):
    """递归查找所有符合条件的脑电文件"""
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return files

def load_raw(path: Path):
    """根据文件后缀使用不同的 MNE 方法读取脑电数据"""
    if path.suffix == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    elif path.suffix == ".fif":
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    elif path.suffix == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    elif path.suffix == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    else:
        raise ValueError(f"暂不支持的文件格式: {path}")
    return raw

def parse_labels(participants_tsv: Path, label_column: str | None = None):
    """从 participants.tsv 文件中解析被试的标签信息"""
    df = pd.read_csv(participants_tsv, sep="\t")
    if "participant_id" not in df.columns:
        raise ValueError("participants.tsv 缺少 participant_id 列")

    # 如果没指定标签列，尝试从常见的名称中搜索
    if label_column is None:
        for cand in ["group", "Group", "diagnosis", "Diagnosis", "dx", "DX", "label", "Label"]:
            if cand in df.columns:
                label_column = cand
                break
    if label_column is None:
        raise ValueError("无法推断标签列，请使用 --label-column 手动指定")

    labels = dict(zip(df["participant_id"].astype(str), df[label_column].astype(str)))
    return labels, label_column

def subject_id_from_path(path: Path):
    """从文件路径中提取符合 BIDS 标准的被试 ID (sub-XXXX)"""
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    return None

def build_subject_dataset(data_root: Path, use_derivatives: bool, label_column: str | None,
                          window_sec: float, step_sec: float):
    """构建完整的被试特征数据集"""
    participants_tsv = data_root / "participants.tsv"
    if not participants_tsv.exists():
        raise FileNotFoundError(f"找不到标签文件: {participants_tsv}")

    labels, inferred_col = parse_labels(participants_tsv, label_column)

    # 确定搜索范围：原始数据还是预处理后的衍生数据
    search_root = data_root / "derivatives" if use_derivatives else data_root
    files = find_eeg_files(search_root)
    if not files:
        raise FileNotFoundError(f"在 {search_root} 下未找到任何 EEG 文件")

    X, y, subjects = [], [], []
    feature_names = None

    for f in files:
        sid = subject_id_from_path(f)
        if sid is None or sid not in labels:
            continue

        raw = load_raw(f)
        raw.pick("eeg")
        sfreq = raw.info["sfreq"]
        data = raw.get_data()

        win = int(window_sec * sfreq)
        step = int(step_sec * sfreq)
        if win <= 0 or step <= 0 or data.shape[1] < win:
            continue

        window_features = []
        for start in range(0, data.shape[1] - win + 1, step):
            window = data[:, start:start + win]
            feats, names = extract_window_features(window, sfreq)
            window_features.append(feats)
            if feature_names is None:
                # 生成均值和标准差的特征名称
                feature_names = [f"mean_{n}" for n in names] + [f"std_{n}" for n in names]

        if not window_features:
            continue

        # 聚合窗口特征为被试特征
        subj_feat = aggregate_subject_features(window_features)
        X.append(subj_feat)
        y.append(labels[sid])
        subjects.append(sid)

    return np.asarray(X), np.asarray(y), subjects, feature_names, inferred_col

def get_model(name: str):
    """获取指定的分类模型"""
    name = name.lower()
    if name == "svm":
        return SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    if name == "lr":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "stacking":
        # 模型堆叠 (Stacking)：结合 SVM 和 随机森林 的优势
        estimators = [
            ("svm", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced")),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5
        )
    raise ValueError("不支持的模型名称。请选择: svm, rf, lr, stacking")

def main():
    parser = argparse.ArgumentParser(description="脑电分类训练脚本")
    parser.add_argument("--data-root", type=Path, default=Path("../ds004504"), help="数据集根目录")
    parser.add_argument("--use-derivatives", action="store_true", default=True, help="是否使用衍生数据")
    parser.add_argument("--label-column", default=None, help="标签列名")
    parser.add_argument("--window-sec", type=float, default=2.0, help="分窗长度(秒)")
    parser.add_argument("--step-sec", type=float, default=1.0, help="步长(秒)")
    parser.add_argument("--model", type=str, default="svm", help="选择模型")
    parser.add_argument("--cv", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--report", type=Path, default=Path("report.json"), help="结果保存路径")
    args = parser.parse_args()

    # 1. 构建数据集
    X, y, subjects, feat_names, inferred_col = build_subject_dataset(
        args.data_root, args.use_derivatives, args.label_column,
        args.window_sec, args.step_sec
    )

    classes = sorted(set(y.tolist()))
    
    # 2. 初始化分类流水线
    model = get_model(args.model)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])

    # 3. 运行交叉验证
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    accs, f1s = [], []
    y_true_all, y_pred_all = [], []

    print(f"开始对 {len(subjects)} 个被试进行 {args.cv} 折交叉验证...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(preds)
        accs.append(accuracy_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds, average="macro"))
        print(f"Fold {fold+1} 完成")

    # 4. 生成报表并保存
    report = {
        "n_subjects": int(X.shape[0]),
        "label_column": inferred_col,
        "classes": classes,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all, labels=classes).tolist(),
        "classification_report": classification_report(y_true_all, y_pred_all, labels=classes, output_dict=True),
    }

    args.report.write_text(json.dumps(report, indent=2))
    print("\n--- 训练结束 ---")
    print(f"平均准确率: {np.mean(accs):.4f}")
    print(f"平均 Macro-F1: {np.mean(f1s):.4f}")

if __name__ == "__main__":
    main()
