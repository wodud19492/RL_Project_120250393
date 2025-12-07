import pandas as pd
import numpy as np
import librosa

# 데이터 로드
df = pd.read_csv("GRBAS_dataset.csv")

# 0) duration 계산
def get_duration_sec(path, sr=None):
    try:
        y, sr = librosa.load(path, sr=sr)
        return len(y) / sr
    except Exception:
        return np.nan

df["duration_sec"] = df["wav_path"].apply(get_duration_sec)

# 1) 전체 sample 수
print("Total samples:", len(df))

# 2) vowel별 sample 수
print("\nSamples per vowel_label:")
print(df["vowel_label"].value_counts().sort_index())

# 3) 전체 GRBAS 평균/분산 (SLPall)
grbas_cols = ["SLPall_G", "SLPall_R", "SLPall_B", "SLPall_A", "SLPall_S"]
print("\nGRBAS mean (SLPall):")
print(df[grbas_cols].mean())
print("\nGRBAS variance (SLPall):")
print(df[grbas_cols].var())

# 4) 전체 audio length (초)
total_duration = df["duration_sec"].sum()
mean_duration  = df["duration_sec"].mean()
print("\nTotal audio length (sec):", total_duration)
print("Mean audio length  (sec):", mean_duration)

# 5) vowel별 audio length (초)
print("\nAudio length per vowel_label (sec):")
vowel_dur = df.groupby("vowel_label")["duration_sec"].agg(["count", "sum", "mean"])
print(vowel_dur)
