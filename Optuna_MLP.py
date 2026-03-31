# Tuning Optuna cho MLP
import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tensorflow import keras
from tensorflow.keras import layers
import time
import json 

# ================= LOAD DATA =================
data = np.load('data_for_tuning_mlp.npz')
X = data['Xtrain'].astype(np.float32)
y = data['Ytrain'].astype(np.int32)
sample_weights = data['sample_weights']
start_time = time.time()

# ================= OBJECTIVE =================
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    units_layer1 = trial.suggest_int('units_layer1', 32, 512, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    units_layer2 = trial.suggest_int('units_layer2', 16, 256, log=True) if num_layers >= 2 else 0
    units_layer3 = trial.suggest_int('units_layer3', 8, 128, log=True) if num_layers >= 3 else 0
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    f1_scores = []

    for tr_idx, val_idx in skf.split(X, y):
        model = keras.Sequential()
        model.add(layers.Dense(units_layer1, activation='relu',
        input_shape=(X.shape[1],)))
        model.add(layers.Dropout(dropout_rate))
        if num_layers >= 2:
            model.add(layers.Dense(units_layer2, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
        if num_layers >= 3:
            model.add(layers.Dense(units_layer3, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(4, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        
        history = model.fit(
            X[tr_idx],
            y[tr_idx],
            epochs=100,
            batch_size=batch_size,
            validation_data=(X[val_idx], y[val_idx]),
            sample_weight=sample_weights[tr_idx],
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=20,
            restore_best_weights=True)]) 
        
        preds = model.predict(X[val_idx])
        preds_class = np.argmax(preds, axis=1)
        f1 = f1_score(y[val_idx], preds_class, average='macro')
        f1_scores.append(f1)

    return np.mean(f1_scores)

# ================= STUDY =================
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.HyperbandPruner(),
    sampler=optuna.samplers.TPESampler(seed=42),
    storage='sqlite:///optuna_mlp.db',
    study_name='mlp_engine_fault',
    load_if_exists=True)
 
#Callback in log chi tiết từng trial
def on_trial_finish(study, trial):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[I {timestamp}] Trial {trial.number} finished with value: "
          f"{trial.value:.7f} and parameters: {trial.params}. "
          f"Best is trial {study.best_trial.number} with value: {study.best_value:.7f}.")

study.optimize(
        objective,
        n_trials=30,
        timeout=3600*1,
        callbacks=[on_trial_finish],
        show_progress_bar=True) 

#================= SAVE =================

total_time = time.time() - start_time
result = {
    'best_macro_f1': float(study.best_value),
    'best_params': study.best_params,
    'n_trials': len(study.trials),
    'time_seconds': float(total_time),
    'datetime_finished': time.strftime('%Y-%m-%d %H:%M:%S')} 

with open('best_params_mlp.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print("\nMLP OPTUNA TUNING DONE")
print(f"Best Macro F1 : {study.best_value:.5f}")
print(f"Trials : {len(study.trials)}")
print("Saved to best_params_mlp.json")