# tune_rf.py - Tuning Optuna cho Random Forest
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
import json
import time

# ================= LOAD DATA =================
data = np.load('data_for_tuning_RF.npz')
X = data['Xtrain'].astype(np.float32)
y = data['Ytrain'].astype(np.int32)
sample_weights = data['sample_weights']
start_time = time.time()

# ================= OBJECTIVE =================
def objective(trial):
    params = {
    'n_estimators': trial.suggest_int('n_estimators', 100, 2000, log=True),  
        # số cây trong rừng, càng nhiều càng chính xác nhưng chậm, log=True vì thường tăng theo cấp số nhân
    'max_depth': trial.suggest_int('max_depth', 5, 30),  
        # độ sâu tối đa của mỗi cây, hạn chế overfitting
    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),  
        # số mẫu tối thiểu để chia một node, nhỏ → cây phức tạp hơn
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),  
        # số mẫu tối thiểu ở một lá, nhỏ → dễ overfit, lớn → cây gọn hơn
    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),  
        # số feature thử cho mỗi split, sqrt/log2 → giảm correlation giữa cây
    'class_weight': 'balanced',  
        # tự gán trọng số cho lớp thiểu số, tránh bias class
    'random_state': 42,  
        # đảm bảo kết quả lặp lại
    'n_jobs': -1  
        # chạy song song nhiều CPU để nhanh hơn
    }
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    f1_scores = []

    for tr_idx, val_idx in skf.split(X, y):
        clf = RandomForestClassifier(**params)
        clf.fit(
            X[tr_idx],
            y[tr_idx],
            sample_weight=sample_weights[tr_idx]
        ) 
        preds = clf.predict(X[val_idx])
        f1 = f1_score(y[val_idx], preds, average='macro')
        f1_scores.append(f1)

    return np.mean(f1_scores)

# ================= STUDY =================
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.HyperbandPruner(),
    sampler=optuna.samplers.TPESampler(seed=42),
    storage='sqlite:///optuna_rf.db',
    study_name='rf_engine_fault',
    load_if_exists=True
) 

# Callback gọi khi 1 trial kết thúc
def on_trial_finish(study, trial):
    # Lấy thời gian hiện tại để log
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # In thông tin trial vừa chạy xong
    print(
        f"[I {timestamp}] Trial {trial.number} finished with value: "
        f"{trial.value:.7f} and parameters: {trial.params}. "
        f"Best is trial {study.best_trial.number} with value: {study.best_value:.7f}."
    )
    
# Thực hiện tối ưu hyperparameter
study.optimize(objective, n_trials=30,
            timeout=3600*0.5, # 30 phút
            callbacks=[on_trial_finish],
            show_progress_bar=True)
     

#================= SAVE =================
total_time = time.time() - start_time
result = {
    'best_macro_f1': float(study.best_value),
    'best_params': study.best_params,
    'n_trials': len(study.trials),
    'time_seconds': float(total_time),
    'datetime_finished': time.strftime('%Y-%m-%d %H:%M:%S')
} 

with open('best_params_rf.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
    
print("\nRANDOM FOREST OPTUNA TUNING DONE")
print(f"Best Macro F1 : {study.best_value:.5f}")
print(f"Trials : {len(study.trials)}")
print("Saved to best_params_rf.json")