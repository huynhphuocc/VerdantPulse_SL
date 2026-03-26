# TUNE RANDOM FOREST - PREPARE DATA
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

# 1. Đọc CSV 
df = pd.read_csv("EngineFaultDB_Final.csv")  
print("CSV loaded successfully.")
# 2. Tách X và y
y = df["Fault"].values  # label
X = df.drop("Fault", axis=1).values  # input

#  3. Ép kiểu dữ liệu 
X = X.astype(np.float32)
y = y.astype(np.int32)

#  4. Tạo sample weight 
sample_weights = compute_sample_weight(class_weight="balanced", y=y)
    # wi= số mẫu tổng cộng / (số lớp * số mẫu của lớp i)

#  5. Lưu thành file .npz 
np.savez("data_for_tuning_RF.npz",
         Xtrain=X,
         Ytrain=y,
         sample_weights=sample_weights) 
        #1 mảng cùng số dòng với X/y, mỗi mẫu có 1 trọng số để xử lý mất cân bằng lớp

print("Created file data_for_tuning_RF.npz")