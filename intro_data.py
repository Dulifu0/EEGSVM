import mne
from pathlib import Path
import numpy as np

def intro_sub01_data():
    """
    这个脚本用于展示 sub-001 被试的脑电数据结构。
    它会告诉你数据有多少通道、采样率是多少、以及数据矩阵的具体形状。
    """
    
    # 1. 确定文件路径 (指向 derivatives 文件夹下的预处理数据)
    # 使用相对路径计算，确保灵活性,你用絕對路徑也行
    BASE_DIR = Path(__file__).resolve().parent.parent
    file_path = BASE_DIR / "ds004504" / "derivatives" / "sub-001" / "eeg" / "sub-001_task-eyesclosed_eeg.set"
    
    if not file_path.exists():
        print(f"错误：找不到文件 {file_path}")
        return

    print(f"--- 正在读取文件: {file_path.name} ---")
    
    # 2. 读取脑电数据 (preload=True 表示将数据加载到内存中)
    raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
    
    # 新增：直接打印整个 info 对象，查看所有可用的元数据暗号
    print(f"\n[0. 元数据概览 (raw.info)]")
    print(raw.info)
    
    # 3. 提取脑电的基本信息 (Info)
    sfreq = raw.info['sfreq']    # 采样率：每秒采集多少个数据点
    n_channels = len(raw.ch_names) # 通道数：通常为 19
    times = raw.times            # 时间轴数组
    duration = raw.times[-1]     # 信号的总时长（秒）
    
    print(f"\n[1. 基本规格]")
    print(f"通道数量: {n_channels}")
    print(f"通道名称: {raw.ch_names}")
    print(f"采样频率: {sfreq} Hz (即每秒采集 {sfreq} 个数字)")
    print(f"信号时长: {duration:.2f} 秒")
    
    # 4. 观察数据形状 (Shape)
    # 脑电数据在内存中是一个 2D 矩阵: (通道, 时间点)
    data = raw.get_data()
    print(f"\n[2. 数据形状 (Shape)]")
    print(f"矩阵形状: {data.shape}  -->  (19 个通道, {data.shape[1]} 个时间点)")
    
    # 5. 观察数值内容 (Content)
    # 原始数据的单位是伏特 (V)，通常非常小
    print(f"\n[3. 数值示例 (单位: 伏特)]")
    print(f"前 3 个通道的前 5 个采样点:\n{data[:3, :5]}")
    
    # 6. 计算分窗 (Windowing) 的例子
    # 如果我们将数据切成 2 秒一段
    win_pts = int(2.0 * sfreq)
    window_sample = data[:, 0:win_pts]
    print(f"\n[4. 分窗示例]")
    print(f"如果取 2 秒作为一个窗口，它的矩阵形状是: {window_sample.shape}")
    print(f"这意味着模型一次会输入 {window_sample.shape[0] * window_sample.shape[1]} 个原始数值")

    # 7. 统计信息 (了解信号的幅度范围)
    # 1e6 是为了将 伏特 (V) 转换为 微伏 (μV)
    print(f"\n[5. 统计信息 (单位: μV)]")
    print(f"最大电压: {np.max(data)*1e6:.2f} μV")
    print(f"最小电压: {np.min(data)*1e6:.2f} μV")
    print(f"平均电压: {np.mean(data)*1e6:.2f} μV")

if __name__ == "__main__":
    intro_sub01_data()
