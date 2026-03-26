# OPTUNA TUNING LIGHTGBM
import optuna
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import json
import time
# ================= LOAD DATA =================
data = np.load('data_for_tuning_lgbm.npz')
X = data['Xtrain'].astype(np.float32)
y = data['Ytrain'].astype(np.int32)
sample_weights = data['sample_weights']
start_time = time.time()

# ================= OBJECTIVE =================
def objective(trial):
    params = {
    'objective': 'multiclass',       # Bài toán phân loại nhiều lớp (Fault 0→3)
    'num_class': 4,                   # Có 4 lớp cần dự đoán
    'metric': 'multi_logloss',        # Hàm loss dùng để đánh giá model khi train
    'boosting_type': 'gbdt',          # Chọn GBDT (Gradient Boosting Decision Tree)
    
    # Các tham số mà Optuna sẽ tự thử (tuning)
    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True), 
        # tốc độ học, log=True vì tốc độ học thường thay đổi theo cấp số nhân
    'num_leaves': trial.suggest_int('num_leaves', 8, 128, log=True),  
        # số lá tối đa trong một cây → ảnh hưởng độ phức tạp của cây
    'max_depth': trial.suggest_int('max_depth', 3, 12),  
        # độ sâu tối đa của cây, hạn chế overfitting
    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 300),  
        # số mẫu tối thiểu trong một lá → tránh lá quá ít mẫu
    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),  
        # tỉ lệ feature dùng trong mỗi cây → giống như dropout
    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),  
        # tỉ lệ mẫu dùng để train mỗi cây → giảm overfitting
    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),  
        # tần suất bagging, ví dụ 1 = mỗi vòng train lại bagging
    'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),  
        # regularization L1 → giảm overfitting
    'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),  
        # regularization L2 → giảm overfitting
    'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),  
        # chỉ chia cây nếu gain ≥ giá trị này → kiểm soát phân nhánh
    'verbosity': -1,                  # tắt log khi train
    'random_state': 42,               # seed để kết quả có thể lặp lại
    'deterministic': True,            # đảm bảo tính ổn định trên cùng dữ liệu
    }

    # StratifiedKFold: chia dữ liệu train/val theo tỉ lệ lớp giống nhau
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    f1_scores = []  # list lưu kết quả F1-score mỗi fold

    for tr_idx, val_idx in skf.split(X, y):
    # ===== 1. Tạo Dataset cho LGBM =====
    # LGBM cần object Dataset riêng, không train trực tiếp với array
    lgb_train = lgb.Dataset(
        X[tr_idx],
        label=y[tr_idx],
        weight=sample_weights[tr_idx],  # dùng sample weight để cân bằng lớp
        free_raw_data=False               # giữ dữ liệu gốc trong bộ nhớ
    ) 
    lgb_val = lgb.Dataset(
        X[val_idx],
        label=y[val_idx],
        weight=sample_weights[val_idx],  # cân bằng lớp cho validation
        reference=lgb_train,             # tham chiếu train set để tính metric chuẩn
        free_raw_data=False
    ) 

        # ===== 2️. Huấn luyện model =====
        model = lgb.train(
            params,                  # tham số do Optuna gợi ý
            lgb_train,               # train dataset
            num_boost_round=5000,    # số cây tối đa
            valid_sets=[lgb_val],    # validation set dùng cho early stopping
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=150,  # dừng sớm nếu 150 vòng không cải thiện metric
                    verbose=False
                ),
                lgb.log_evaluation(period=0)  # không log từng vòng ra console
            ]
        )

        # ===== 3️. Dự đoán =====
        preds = model.predict(X[val_idx])         # trả về xác suất cho từng lớp
        preds_class = np.argmax(preds, axis=1)    # chọn lớp có xác suất cao nhất

        # ===== 4️. Đánh giá =====
        f1 = f1_score(y[val_idx], preds_class, average='macro')  # macro F1 cho multi-class
        f1_scores.append(f1)

    return np.mean(f1_scores)

# ================= STUDY =================
# Tạo study Optuna để tìm hyperparameters tốt nhất
study = optuna.create_study(
    direction='maximize',                        # mục tiêu: maximize macro F1
    pruner=optuna.pruners.HyperbandPruner(),     # dừng trial sớm nếu không cải thiện (tiết kiệm thời gian)
    sampler=optuna.samplers.TPESampler(seed=42), # Bayesian sampling: TPE
    storage='sqlite:///optuna_lgbm.db',         # lưu study vào SQLite (load lại được nếu dừng giữa chừng)
    study_name='lgbm_engine_fault',             # tên study
    load_if_exists=True                          # load lại study nếu đã tồn tại
) 

# Thực hiện tối ưu hyperparameter
study.optimize(objective, 
               n_trials=60,      # số thử nghiệm tối đa
               timeout=3600*2)   # hoặc dừng sau 2 giờ, cái nào tới trước

# ================= SAVE =================
# Tính tổng thời gian chạy
total_time = time.time() - start_time

# Chuẩn bị kết quả để lưu
result = {
    'best_macro_f1': float(study.best_value),   # macro F1 tốt nhất
    'best_params': study.best_params,           # tham số tương ứng
    'n_trials': len(study.trials),              # số trial đã chạy
    'time_seconds': float(total_time),          # tổng thời gian chạy
    'datetime_finished': time.strftime('%Y-%m-%d %H:%M:%S')  # thời điểm hoàn thành
} 

# Lưu kết quả vào file JSON
with open('best_params_lgbm.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print("\nLIGHTGBM OPTUNA TUNING DONE")
print(f"Best Macro F1 : {study.best_value:.5f}")
print(f"Trials : {len(study.trials)}")
print("Saved to best_params_lgbm.json")