%% TRAIN_RF_OPTUNA
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
    if ~isfile('best_params_rf.json')
        error('File best_params_rf.json không tồn tại. Chạy tune_rf.m trước!');
    end
    json_str = fileread('best_params_rf.json');
    best_result = jsondecode(json_str);
    
    if isfield(best_result, 'best_macro_f1')
        best_value = best_result.best_macro_f1;
    elseif isfield(best_result, 'best_value')
        best_value = best_result.best_value;
    else
        best_value = NaN;
        fprintf('Không tìm thấy field best_macro_f1/best_value (bỏ qua).\n');
    end
    fprintf('Best Macro F1 từ tuning: %.4f\n', best_value);
    if isfield(best_result, 'best_params')
        bp = best_result.best_params;
        disp('Best hyperparameters:');
        disp(bp);
    else
        error('Không tìm thấy field "best_params" trong JSON.');
    end

%% 3. Train model Random Forest
    fprintf('Đang train model Random Forest cuối cùng...\n');
    sklearn_ensemble = py.importlib.import_module('sklearn.ensemble');
    np = py.importlib.import_module('numpy');
    max_features_val = bp.max_features;
    
    if iscell(max_features_val) || isempty(max_features_val) || isa(max_features_val,'py.numpy.ndarray') && double(max_features_val.size) == 0
        max_features_val = 'sqrt';
        fprintf('Cảnh báo: max_features từ JSON là rỗng hoặc array → fallback sang''sqrt''.\n');
    elseif ischar(max_features_val) || isnumeric(max_features_val)
        else
         max_features_val = 'sqrt';
    end
    model = sklearn_ensemble.RandomForestClassifier(pyargs(...
    'n_estimators', py.int(double(bp.n_estimators)), ...
    'max_depth', py.int(double(bp.max_depth)), ...
    'min_samples_split', py.int(double(bp.min_samples_split)), ...
    'min_samples_leaf', py.int(double(bp.min_samples_leaf)), ...
    'max_features', max_features_val, ...
    'class_weight', 'balanced', ...
    'random_state', py.int(42) ...
    ));
    model.fit(np.array(Xtrain), np.array(Ytrain), pyargs('sample_weight', np.array(single(sample_weights_vec))));

%% 4. Dự đoán
Ypred_num = double(model.predict(np.array(Xtest)));
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
    h.Title = 'Random Forest (Optuna Bayesian Optimized)';
    h.XLabel = 'Predicted Class';
    h.YLabel = 'Actual Class';
    h.DiagonalColor = [0.35 0.70 0.40];
    h.OffDiagonalColor = [0.90 0.97 0.92];
    h.FontSize = 14;
    h.GridVisible = 'on';
    annotation('textbox',[0.55 0.05 0.40 0.08], ...
    'String',sprintf('Overall Accuracy = %.3f%% | Macro F1 = %.3f', Accuracy*100, macroF1), ...
    'FontSize',14,'FontWeight','bold', ...
    'BackgroundColor',[0.92 0.98 0.93], 'EdgeColor',[0.35 0.70 0.40], ...
    'HorizontalAlignment','center');
    annotation('textbox',[0.2 0.88 0.6 0.09], ...
    'String','Random Forest - Optuna Optimized 2025', ...
    'FontSize',26,'FontWeight','bold','Color',[0.2 0.5 0.2], ...
    'HorizontalAlignment','center','EdgeColor','none','BackgroundColor','white');
    % LƯU MODEL RANDOM FOREST
    save('model_rf.mat', 'model');
    disp('HOÀN THÀNH! Random Forest đã được tối ưu Bayesian và đánh giá đầy đủ. Model đã lưu tại: model_rf.mat');