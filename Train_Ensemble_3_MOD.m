%% ENSEMBLE 3 MODEL: RF + LGBM + MLP
clc; clear; close all;

%% 1. Load dữ liệu
T = readtable('EngineFaultDB_Final.csv','VariableNamingRule','preserve');
Y_cat = categorical(T{:,1});
Y = grp2idx(Y_cat) - 1;
X = table2array(T(:,2:end));
X = fillmissing(X,'constant',0);
X = normalize(X,'zscore');

rng(42);
cv = cvpartition(Y_cat,'Holdout',0.2,'Stratify',true);
trainIdx = training(cv);
testIdx = test(cv);
Xtrain = X(trainIdx,:); Ytrain = Y(trainIdx);
Xtest = X(testIdx,:); Ytest = Y(testIdx);
Ytrain_cat = Y_cat(trainIdx);
Ytest_cat = Y_cat(testIdx);

%% 2. Sample weights
class_counts_train = histcounts(Ytrain_cat);
class_weights = max(class_counts_train) ./ class_counts_train;
sample_weights_vec = class_weights(Ytrain + 1);

%% 3. Load best params từ JSON (RF, LGBM, MLP)
fprintf('Load best params 3 model...\n');
bp_lgb = jsondecode(fileread('best_params_lgbm.json')).best_params;
bp_rf  = jsondecode(fileread('best_params_rf.json')).best_params;
bp_mlp = jsondecode(fileread('best_params_mlp.json')).best_params;
fprintf('Load params thành công!\n');

%% 4. Train lại 3 model
fprintf('Train lại 3 model...\n');
sklearn_ensemble = py.importlib.import_module('sklearn.ensemble');

% Random Forest
model_rf = sklearn_ensemble.RandomForestClassifier(pyargs( ...
    'n_estimators', py.int(double(bp_rf.n_estimators)), ...
    'max_depth', py.int(double(bp_rf.max_depth)), ...
    'random_state', py.int(42), ...
    'class_weight', 'balanced' ...
));
model_rf.fit(py.numpy.array(single(Xtrain)), py.numpy.array(int64(Ytrain)), ...
    pyargs('sample_weight', py.numpy.array(single(sample_weights_vec))));

% LightGBM
lgb = py.importlib.import_module('lightgbm');
trainSet = lgb.Dataset(py.numpy.array(single(Xtrain)), ...
    py.numpy.array(int64(Ytrain)), weight=py.numpy.array(single(sample_weights_vec)));
evalSet = lgb.Dataset(py.numpy.array(single(Xtest)), py.numpy.array(int64(Ytest)), ...
    reference=trainSet);
params_lgb = py.dict(pyargs( ...
    'objective', 'multiclass', 'num_class', py.int(4), ...
    'learning_rate', double(bp_lgb.learning_rate), ...
    'num_leaves', py.int(double(bp_lgb.num_leaves)), ...
    'max_depth', py.int(double(bp_lgb.max_depth)), ...
    'random_state', py.int(42), ...
    'verbosity', py.int(-1) ...
));
model_lgb = lgb.train(params_lgb, trainSet, py.int(500), pyargs('valid_sets', py.list({evalSet})));

%% MLP (Fixed MATLAB ↔ Keras)
keras = py.importlib.import_module('keras');
layers = keras.layers;
models = keras.models;
optimizers = keras.optimizers;

input_dim = int32(size(Xtrain,2));  % số feature đầu vào

% Khởi tạo Sequential model
model_mlp = models.Sequential();

% --- Layer 1 ---
model_mlp.add(layers.Dense(py.int(double(bp_mlp.units_layer1)), ...
    pyargs('activation','relu','input_shape', py.tuple({py.int(input_dim)}))));
model_mlp.add(layers.Dropout(double(bp_mlp.dropout_rate)));

% --- Layer 2 (nếu có) ---
if double(bp_mlp.num_layers) >= 2
    model_mlp.add(layers.Dense(py.int(double(bp_mlp.units_layer2)), ...
        pyargs('activation','relu')));
    model_mlp.add(layers.Dropout(double(bp_mlp.dropout_rate)));
end

% --- Layer 3 (nếu có) ---
if double(bp_mlp.num_layers) >= 3
    model_mlp.add(layers.Dense(py.int(double(bp_mlp.units_layer3)), ...
        pyargs('activation','relu')));
    model_mlp.add(layers.Dropout(double(bp_mlp.dropout_rate)));
end

% --- Output layer ---
model_mlp.add(layers.Dense(py.int(4), pyargs('activation','softmax')));

% --- Compile model ---
model_mlp.compile(optimizers.Adam(double(bp_mlp.learning_rate)), ...
    'sparse_categorical_crossentropy', metrics=py.list({'accuracy'}));

% --- Train model ---
model_mlp.fit(py.numpy.array(single(Xtrain)), py.numpy.array(int64(Ytrain)), ...
    pyargs('epochs', py.int(100), ...
           'batch_size', py.int(double(bp_mlp.batch_size)), ...
           'validation_data', py.tuple({py.numpy.array(single(Xtest)), py.numpy.array(int64(Ytest))}), ...
           'sample_weight', py.numpy.array(single(sample_weights_vec)), ...
           'verbose', py.int(0)));

fprintf('Train 3 model xong!\n');

%% 5. Dự đoán xác suất
n_samples = size(Xtest,1);
n_classes = 4;
proba_all = zeros(n_samples, n_classes, 3);

proba_all(:,:,1) = double(model_rf.predict_proba(py.numpy.array(single(Xtest))));
proba_all(:,:,2) = double(model_lgb.predict(py.numpy.array(single(Xtest))));
proba_all(:,:,3) = double(model_mlp.predict(py.numpy.array(single(Xtest))));

%% 6. Soft Voting & Blending
proba_soft = mean(proba_all,3);
[~, Ypred_soft] = max(proba_soft, [], 2); Ypred_soft = Ypred_soft - 1;
Ypred_soft_cat = categorical(Ypred_soft+1, 1:4, categories(Y_cat));

weights = [0.35, 0.35, 0.30]; % có thể tùy chỉnh
proba_blend = zeros(n_samples,n_classes);
for m = 1:3
    proba_blend = proba_blend + weights(m) * proba_all(:,:,m);
end
[~, Ypred_blend] = max(proba_blend,[],2); Ypred_blend = Ypred_blend - 1;
Ypred_blend_cat = categorical(Ypred_blend+1,1:4,categories(Y_cat));

%% 7. Đánh giá
fprintf('\n=== ENSEMBLE SOFT VOTING ===\n');
cm_soft = confusionmat(Ytest_cat, Ypred_soft_cat,'Order',categories(Y_cat));
acc_soft = sum(diag(cm_soft))/sum(cm_soft(:));
f1_soft = mean(2*diag(cm_soft)./(sum(cm_soft,2)+sum(cm_soft,1)'+eps));
fprintf('Accuracy: %.4f%% | Macro F1: %.4f\n', acc_soft*100, f1_soft);

fprintf('\n=== ENSEMBLE BLENDING ===\n');
cm_blend = confusionmat(Ytest_cat,Ypred_blend_cat,'Order',categories(Y_cat));
acc_blend = sum(diag(cm_blend))/sum(cm_blend(:));
f1_blend = mean(2*diag(cm_blend)./(sum(cm_blend,2)+sum(cm_blend,1)'+eps));
fprintf('Accuracy: %.4f%% | Macro F1: %.4f\n', acc_blend*100,f1_blend);

disp('HOÀN THÀNH! Ensemble 3 model hoàn tất.');

%% 8. CONFUSION MATRIX (Ensemble)
figure('Position',[300 150 720 620],'Color','w');

% Chọn cm_blend hoặc cm_soft
h = confusionchart(cm_blend, categories(Y_cat), 'Normalization','absolute');

h.Title = 'Ensemble (Optuna Bayesian Optimized)';
h.XLabel = 'Predicted Class';
h.YLabel = 'Actual Class';
h.DiagonalColor = [0.35 0.70 0.40];
h.OffDiagonalColor = [0.90 0.97 0.92];
h.FontSize = 14;
h.GridVisible = 'on';

annotation('textbox',[0.55 0.05 0.40 0.08], ...
    'String',sprintf('Overall Accuracy = %.3f%% | Macro F1 = %.3f', acc_blend*100, f1_blend), ...
    'FontSize',13,'FontWeight','bold', ...
    'BackgroundColor',[0.92 0.98 0.93], 'EdgeColor',[0.35 0.70 0.40], ...
    'HorizontalAlignment','center');
