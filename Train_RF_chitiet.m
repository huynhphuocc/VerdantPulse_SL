%% ================= TRAIN_RF_OPTUNA =================
clc; clear; close all;

%% 1. Load dữ liệu
% Đọc bảng dữ liệu CSV gốc
T = readtable('EngineFaultDB_Final.csv','VariableNamingRule','preserve');

% Chuyển cột Fault đầu tiên thành categorical và sau đó thành nhãn số từ 0
Y_cat = categorical(T{:,1});          % Y_cat = ['Fault0','Fault1', ...]
Y = grp2idx(Y_cat) - 1;               % Nhãn số 0,1,2,3 (Random Forest dùng nhãn số)

% Chuyển các cột còn lại thành ma trận số
X = table2array(T(:,2:end));

% Xử lý missing value: điền bằng 0
X = fillmissing(X,'constant',0);

% Chuẩn hóa Z-score (mean=0, std=1) giúp thuật toán hội tụ tốt hơn
X = normalize(X,'zscore');

% Cố định seed để tái lập kết quả
rng(42);

% Tách tập train/test 80:20, giữ tỉ lệ lớp (stratify)
cv = cvpartition(Y_cat,'Holdout',0.2,'Stratify',true);
trainIdx = training(cv);
testIdx = test(cv);
Xtrain = X(trainIdx,:); Ytrain = Y(trainIdx);
Xtest = X(testIdx,:); Ytest = Y(testIdx);
Ytrain_cat = Y_cat(trainIdx);
Ytest_cat = Y_cat(testIdx);

% Tạo vector trọng số lớp để xử lý mất cân bằng lớp
class_counts_train = histcounts(Ytrain_cat);        % Số lượng mỗi lớp trong tập train
class_weights = max(class_counts_train) ./ class_counts_train; % weight = max_count / count
sample_weights_vec = class_weights(Ytrain + 1);     % gán weight theo nhãn

%% 2. Đọc best params từ tuning Optuna
% File JSON lưu best hyperparameters sau khi Optuna tuning
if ~isfile('best_params_rf.json')
    error('File best_params_rf.json không tồn tại. Chạy tune_rf.m trước!');
end

% Đọc JSON vào struct
json_str = fileread('best_params_rf.json');
best_result = jsondecode(json_str);

% Lấy giá trị F1 macro tốt nhất
if isfield(best_result, 'best_macro_f1')
    best_value = best_result.best_macro_f1;
elseif isfield(best_result, 'best_value')
    best_value = best_result.best_value;
else
    best_value = NaN;
    fprintf('Không tìm thấy field best_macro_f1/best_value (bỏ qua).\n');
end
fprintf('Best Macro F1 từ tuning: %.4f\n', best_value);

% Lấy các tham số tốt nhất
if isfield(best_result, 'best_params')
    bp = best_result.best_params;
    disp('Best hyperparameters:');
    disp(bp);
else
    error('Không tìm thấy field "best_params" trong JSON.');
end

%% ================= 3. Train Random Forest =================
fprintf('Đang train model Random Forest cuối cùng...\n');

% Import Python module sklearn.ensemble để dùng RandomForestClassifier
sklearn_ensemble = py.importlib.import_module('sklearn.ensemble');
np = py.importlib.import_module('numpy');

% Lấy giá trị max_features từ best params JSON
max_features_val = bp.max_features;

% Kiểm tra giá trị max_features: nếu rỗng hoặc không hợp lệ → fallback sang 'sqrt'
if iscell(max_features_val) || isempty(max_features_val) || ...
   (isa(max_features_val,'py.numpy.ndarray') && double(max_features_val.size) == 0)
    max_features_val = 'sqrt';
    fprintf('Cảnh báo: max_features từ JSON là rỗng hoặc array → fallback sang ''sqrt''.\n');
elseif ischar(max_features_val) || isnumeric(max_features_val)
    % hợp lệ, dùng luôn
else
    max_features_val = 'sqrt'; % fallback
end

% Khởi tạo model Random Forest với các tham số từ Optuna
model = sklearn_ensemble.RandomForestClassifier(pyargs(...
    'n_estimators', py.int(double(bp.n_estimators)), ...
    'max_depth', py.int(double(bp.max_depth)), ...
    'min_samples_split', py.int(double(bp.min_samples_split)), ...
    'min_samples_leaf', py.int(double(bp.min_samples_leaf)), ...
    'max_features', max_features_val, ...
    'class_weight', 'balanced', ...  % để xử lý mất cân bằng lớp
    'random_state', py.int(42) ...
    ));

% Fit model với train set + sample weights
model.fit(np.array(Xtrain), np.array(Ytrain), pyargs('sample_weight', np.array(single(sample_weights_vec))));

%% ================= 4. Dự đoán =================
Ypred_num = double(model.predict(np.array(Xtest)));    % nhãn số dự đoán
Ypred_cat = categorical(Ypred_num + 1, 1:4, categories(Y_cat)); % chuyển thành categorical

%% ================= 5. Tính toán chỉ số đánh giá =================
classes = categories(Y_cat);
cm = confusionmat(Ytest_cat, Ypred_cat, 'Order', classes);  % confusion matrix

% Tính TP, FP, FN
TP = diag(cm);
FP = sum(cm,1)' - TP;
FN = sum(cm,2) - TP;

% Precision, Recall, F1
Precision = TP ./ (TP + FP + eps);
Recall = TP ./ (TP + FN + eps);
F1 = 2*Precision.*Recall ./ (Precision + Recall + eps);
Accuracy = sum(TP)/sum(cm(:));

% Macro metrics (trung bình từng lớp)
macroPrec = mean(Precision);
macroRec = mean(Recall);
macroF1 = mean(F1);

%% ================= 6. Hiển thị kết quả =================
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════╗\n');
fprintf('║ AUTOMOTIVE DIAGNOSIS RESULT                               ║\n');
fprintf('║ EngineFaultDB – Random Forest (Optuna Optimized)          ║\n');
fprintf('╠══════════════════════════════════════════════════════════╣\n');
fprintf('║ Overall Accuracy │ %6.3f%% ║\n', Accuracy*100);
fprintf('║ Macro Precision │ %6.3f ║\n', macroPrec);
fprintf('║ Macro Recall    │ %6.3f ║\n', macroRec);
fprintf('║ Macro F1-Score  │ %6.3f ║\n', macroF1);
fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

% Bảng chi tiết từng lớp
Results = table(classes, Precision, Recall, F1, ...
    'VariableNames',{'Fault_Type','Precision','Recall','F1_Score'});
disp('Chi tiết từng loại lỗi (Fault 0 → 3):')
disp(Results)

%% ================= 7. Confusion Matrix =================
figure('Position',[300 150 720 620],'Color','w');
h = confusionchart(cm, classes, 'Normalization','absolute');
h.Title = 'Random Forest (Optuna Bayesian Optimized)';
h.XLabel = 'Predicted Class';
h.YLabel = 'Actual Class';
h.DiagonalColor = [0.35 0.70 0.40];
h.OffDiagonalColor = [0.90 0.97 0.92];
h.FontSize = 14;
h.GridVisible = 'on';

% Annotation thông tin Accuracy & F1
annotation('textbox',[0.55 0.05 0.40 0.08], ...
    'String',sprintf('Overall Accuracy = %.3f%% | Macro F1 = %.3f', Accuracy*100, macroF1), ...
    'FontSize',14,'FontWeight','bold', ...
    'BackgroundColor',[0.92 0.98 0.93], 'EdgeColor',[0.35 0.70 0.40], ...
    'HorizontalAlignment','center');

%% ================= 8. Lưu model =================
save('model_rf.mat', 'model');
disp('HOÀN THÀNH! Random Forest đã được tối ưu Bayesian và đánh giá đầy đủ. Model đã lưu tại: model_rf.mat');