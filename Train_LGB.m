%% TRAIN_LIGHTGBM_OPTUNA
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
    class_counts_train = histcounts(Ytrain_cat);
    class_weights = max(class_counts_train) ./ class_counts_train;
    sample_weights_vec = class_weights(Ytrain + 1);
%% 2. Đọc best params từ tuning
filePath = 'best_params_lgbm.json';

if ~isfile(filePath)
    error('File best_params_lgbm.json không tồn tại. Chạy tune_lightgbm.m trước!');
end

json_str = fileread(filePath);
best_result = jsondecode(json_str);
best_params = best_result.best_params;
best_value = best_result.best_macro_f1;
    
    fprintf('=== KẾT QUẢ TUNING OPTUNA (LightGBM) ===\n');
    fprintf('Best Macro F1 từ tuning: %.4f\n', best_value);
    disp('Best hyperparameters:');
    disp(best_params);

%% 3. Train model LightGBM cuối cùng với best params
fprintf('Đang train model LightGBM cuối cùng...\n');

% model = py.lightgbm.LGBMClassifier(...
%     objective='multiclass', ...
%     num_class=py.int(4), ...
%     boosting_type='gbdt', ...
%     random_state=py.int(42), ...
%     verbosity=py.int(-1), ...
%     max_depth=py.int(double(best_params.max_depth)), ...
%     learning_rate=double(best_params.learning_rate), ...
%     bagging_fraction = double(best_params.bagging_fraction),    ...
%     colsample_bytree=double(best_params.colsample_bytree), ...
%     min_child_weight=py.int(double(best_params.min_child_weight)), ...
%     min_split_gain=double(best_params.min_split_gain), ...
%     reg_alpha=double(best_params.reg_alpha), ...
%     reg_lambda=double(best_params.reg_lambda), ...
%     num_leaves=py.int(double(best_params.num_leaves)), ...
%     early_stopping_rounds=py.int(200)); % <--- Đặt ở đây, trong constructor100

model = py.lightgbm.LGBMClassifier(pyargs( ...
    'objective','multiclass', ...
    'num_class',py.int(4), ...
    'boosting_type','gbdt', ...
    'random_state',py.int(42), ...
    'verbosity',py.int(-1), ...
    'max_depth', py.int(double(best_params.max_depth)), ...
    'learning_rate', double(best_params.learning_rate), ...
    'num_leaves', py.int(double(best_params.num_leaves)), ...
    'min_child_samples', py.int(double(best_params.min_data_in_leaf)), ...
    'feature_fraction', double(best_params.feature_fraction), ...
    'bagging_fraction', double(best_params.bagging_fraction), ...
    'bagging_freq', py.int(double(best_params.bagging_freq)), ...
    'reg_alpha', double(best_params.lambda_l1), ...
    'reg_lambda', double(best_params.lambda_l2), ...
    'min_split_gain', double(best_params.min_gain_to_split) ...
));
model.fit(...
    py.numpy.array(Xtrain), py.numpy.array(Ytrain), ...
    sample_weight=py.numpy.array(single(sample_weights_vec)), ...
    eval_set=py.list({py.tuple({py.numpy.array(Xtest), py.numpy.array(Ytest)})}));

%% 4. Dự đoán
    Ypred_num = double(model.predict(py.numpy.array(Xtest)));
    Ypred_cat = categorical(Ypred_num + 1, 1:4, categories(Y_cat));

%% 5. Tính toán chỉ số đánh giá
    classes = categories(Y_cat);
    cm = confusionmat(Ytest_cat, Ypred_cat, 'Order', classes);
    TP = diag(cm);
    FP = sum(cm,1)' - TP;
    FN = sum(cm,2) - TP;
    Precision = TP ./ (TP + FP + eps);
    Recall = TP ./ (TP + FN + eps);
    F1 = 2*Precision.*Recall ./ (Precision + Recall + eps);
    Accuracy = sum(TP)/sum(cm(:));
    macroPrec = mean(Precision);
    macroRec = mean(Recall);
    macroF1 = mean(F1);
    best_iter = double(model.best_iteration_) + 1;

%% 6. HIỂN THỊ BẢNG KẾT QUẢ
    fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════╗\n');
fprintf('║ AUTOMOTIVE DIAGNOSIS RESULT                      ║\n');
fprintf('║ EngineFaultDB – LightGBM (Optuna Optimized)             ║\n');
fprintf('╠══════════════════════════════════════════════════════════╣\n');
fprintf('║ Overall Accuracy │ %6.3f%% ║\n', Accuracy*100);
fprintf('║ Macro Precision │ %6.3f ║\n', macroPrec);
fprintf('║ Macro Recall    │ %6.3f ║\n', macroRec);
fprintf('║ Macro F1-Score  │ %6.3f ║\n', macroF1);
fprintf('║ Best iteration  │ %4d ║\n', best_iter);
fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

    Results = table(classes, Precision, Recall, F1, ...
    'VariableNames',{'Fault_Type','Precision','Recall','F1_Score'});
    disp('Chi tiết từng loại lỗi (Fault 0 → 3):')
    disp(Results)

%% 7. CONFUSION MATRIX

figure('Position',[300 150 720 620],'Color','w');
h = confusionchart(cm, classes, 'Normalization','absolute');
h.Title = 'LightGBM (Optuna Bayesian Optimized)';
h.XLabel = 'Predicted Class';
h.YLabel = 'Actual Class';
h.DiagonalColor = [0.35 0.70 0.40];
h.OffDiagonalColor = [0.90 0.97 0.92];
h.FontSize = 14;
h.GridVisible = 'on';
annotation('textbox',[0.55 0.05 0.40 0.08], ...
'String',sprintf('Overall Accuracy = %.3f%% | Macro F1 = %.3f', Accuracy*100, macroF1), ...
'FontSize',13,'FontWeight','bold', ...
'BackgroundColor',[0.92 0.98 0.93], 'EdgeColor',[0.35 0.70 0.40], ...
'HorizontalAlignment','center');
disp('HOÀN THÀNH! LightGBM đã được tối ưu Bayesian và đánh giá đầy đủ.');

%% 8. FEATURE IMPORTANCE (LightGBM)

% Lấy importance từ model Python
importance = double(model.feature_importances_);
% Lấy tên feature (cột trong bảng, bỏ cột Fault)
feature_names = T.Properties.VariableNames(2:end);
% Sắp xếp giảm dần
[importance_sorted, idx] = sort(importance, 'descend');
feature_sorted = feature_names(idx);
% Vẽ biểu đồ cột
figure('Color','w');
bar(importance_sorted);

xticks(1:length(feature_sorted));
xticklabels(feature_sorted);
xtickangle(45);
ylabel('Feature Importance');
title('LightGBM Feature Importance');
grid on;