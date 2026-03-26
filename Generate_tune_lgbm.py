# TUNE LIGHTGBM - PREPARE DATA
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

# ===== 1. Đọc dữ liệu =====
df = pd.read_csv("EngineFaultDB_Final.csv")  
# ===== 2. Tách X và y =====
y = df["Fault"].values  # label
X = df.drop("Fault", axis=1).values  # input

# ===== 3. Ép kiểu dữ liệu =====
X = X.astype(np.float32)
y = y.astype(np.int32)

# ===== 4. Tạo sample weight (quan trọng) =====
sample_weights = compute_sample_weight(class_weight="balanced", y=y)

# ===== 5. Lưu thành file .npz =====
np.savez("data_for_tuning_lgbm.npz",
         Xtrain=X,
         Ytrain=y,
         sample_weights=sample_weights)

print("Created file data_for_tuning_lgbm.npz")