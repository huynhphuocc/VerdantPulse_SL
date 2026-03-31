# ===============================
# Generate .npz cho MLP (Optuna-ready)
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split

# ===== 1. Đọc dữ liệu =====
# CSV chứa dữ liệu cảm biến + cột 'Fault' (0,1,2,3)
df = pd.read_csv("EngineFaultDB_Final.csv")  

# ===== 2. Tách X và y =====
y = df["Fault"].values.reshape(-1,1)  # reshape thành 2D cho encoder nếu cần
X = df.drop("Fault", axis=1).values   # tất cả cột còn lại là features

# ===== 3. Xử lý missing =====
# Thay NaN bằng 0 (có thể dùng trung bình, median, tùy quyết định)
X = np.nan_to_num(X, nan=0.0)

# ===== 4. Chuẩn hóa dữ liệu (rất quan trọng cho MLP) =====
# StandardScaler chuẩn hóa dữ liệu về mean=0, std=1 để MLP học ổn định
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===== 5. One-hot encode label =====
# Không cần cho Optuna MLP nếu dùng sparse_categorical_crossentropy
# nhưng để tham khảo học thuật/visualize có thể dùng OneHot
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

# ===== 6. Chia train/test =====
# stratify=y để giữ nguyên tỷ lệ các lớp Fault trong train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,             # lưu y dạng số nguyên để Optuna dùng sparse_categorical_crossentropy
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===== 7. Tính sample weight =====
# Dùng để cân bằng lớp khi train MLP
sample_weights = compute_sample_weight(class_weight="balanced", y=y.flatten())
sw_train = sample_weights[:len(X_train)]  # weight cho train
sw_test = sample_weights[len(X_train):]   # weight cho test (không dùng trong Optuna, lưu để tham khảo)

# ===== 8. Lưu file .npz cho Optuna =====
# Chú ý key trùng với code Optuna: Xtrain, Ytrain, sample_weights
np.savez_compressed(
    "data_for_tuning_mlp.npz",
    Xtrain=X_train,          # features train
    Ytrain=y_train,          # label train
    sample_weights=sw_train  # sample weight train
)

print("Created file 'data_for_tuning_mlp.npz' for Optuna MLP")

