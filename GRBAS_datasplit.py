import pandas as pd
import numpy as np

df = pd.read_csv("GRBAS_dataset.csv")  # 네가 만든 CSV

# patient 단위로 split
patients = df["patient_ID"].unique()
np.random.seed(42)
np.random.shuffle(patients)

n = len(patients)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)
# 나머지는 test

train_pat = patients[:n_train]
val_pat   = patients[n_train:n_train+n_val]
test_pat  = patients[n_train+n_val:]

train_df = df[df["patient_ID"].isin(train_pat)]
val_df   = df[df["patient_ID"].isin(val_pat)]
test_df  = df[df["patient_ID"].isin(test_pat)]

train_df.to_csv("GRBAS_train.csv", index=False)
val_df.to_csv("GRBAS_val.csv", index=False)
test_df.to_csv("GRBAS_test.csv", index=False)
