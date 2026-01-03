clear; clc; close all;

%% Step 0: CONFIGURATIONS
% General
dataset_name = 'pose';        % 'data', 'pose', or 'illumination'
task = 1;                     % 1 = subject classification, 2 = expression classification (only for data.mat)
train_ratio_task_2 = 0.70;    % training data ratio for task 1
train_ratio_task_1 = 0.8;     % training data ratio for task 2
classifier = 'svm';           % options: 'bayes', 'knn', 'svm', 'boosted_svm'

% KNN parameters
k = 1;                        % number of neighbors for k-NN
% Kernel SVM parameters
kernel_type = 'rbf';          % Choose kernel: 'rbf' or 'poly'
% Boosted SVM parameters
T = 10;                       % number of boosting rounds


if strcmp(dataset_name, 'pose') || strcmp(classifier, 'illumination') 
    task = 1;
end
if strcmp(classifier, 'svm') || strcmp(classifier, 'boosted_svm')
    dataset_name = 'data'; 
    task = 2;
end

% Dimensionality Projection (MDA & PCA)
proj_mode = 'pca'; % Select which project method, 'mda' or 'pca'.
m_mda_vals = [1,10,20];        % MDA dims for DATA, POSE, ILLUMINATION datasets, respectively.
m_pca_vals = [20,10,20];        % PCA dims for DATA, POSE, ILLUMINATION datasets, respectively.

%% Step 1: Load and reshape data
switch dataset_name
    case 'data'
        load('data.mat');  % loads variable: face
        raw = face;        % 24x21x600
        [h, w, N] = size(raw);
        num_subjects = 200;
        images_per_subject = 3;
        D = h * w;
        if task == 1
            X = reshape(raw, D, N)';  % size: 600 x 504
            labels = repelem(1:num_subjects, images_per_subject)';
        elseif task == 2
            % Use only neutral (index 1) and expression (index 2)
            X = zeros(2*num_subjects, D);
            labels = zeros(2*num_subjects, 1);  % 0 = neutral, 1 = expression
            for n = 1:num_subjects
                X(2*n-1, :) = reshape(raw(:, :, 3*n - 2), 1, []);  % neutral
                X(2*n, :)   = reshape(raw(:, :, 3*n - 1), 1, []);  % expression
                labels(2*n-1) = 1;
                labels(2*n) = 2;
            end
            images_per_subject = 2;
        end
        m_mda = m_mda_vals(1);
        m_pca = m_pca_vals(1);


    case 'pose'
        load('pose.mat');  % loads variable: pose
        raw = pose;
        [h, w, I, S] = size(raw);
        num_subjects = S;
        images_per_subject = I;
        D = h * w;
        X = reshape(raw, D, I*S)';  % size: 884 x 1920
        labels = repelem(1:num_subjects, images_per_subject)';
        m_mda = m_mda_vals(2);
        m_pca = m_pca_vals(2);

    case 'illumination'
        load('illumination.mat');  % loads variable: illum
        raw = illum;
        [D, I, S] = size(raw);
        num_subjects = S;
        images_per_subject = I;
        X = zeros(I*S, D);
        for s = 1:S
            for i = 1:I
                X((s-1)*I + i, :) = raw(:, i, s)';
            end
        end
        labels = repelem(1:num_subjects, images_per_subject)';
        m_mda = m_mda_vals(3);
        m_pca = m_pca_vals(3);
end

%% Step 2: Normalize
X = double(X);
X = X - mean(X, 1);
X = X ./ (std(X, 0, 1) + 1e-10);  % avoid divide-by-zero

%% Step 3: Split into training/testing sets 

% Task 2 data split
train_per_class = train_ratio_task_2*num_subjects;
if task == 2 && strcmp(dataset_name, 'data')
    neutral_idx = find(labels == 1);  % 200 indices
    expr_idx    = find(labels == 2);  % 200 indices

    % Randomize within each class
    neutral_idx = neutral_idx(randperm(length(neutral_idx)));
    expr_idx    = expr_idx(randperm(length(expr_idx)));

    % Pick train and test samples per class
    train_idx = [neutral_idx(1:train_per_class); expr_idx(1:train_per_class)];
    test_idx  = [neutral_idx(train_per_class+1:num_subjects); expr_idx(train_per_class+1:num_subjects)];

    % Shuffle to mix classes
    train_idx = train_idx(randperm(length(train_idx)));
    test_idx  = test_idx(randperm(length(test_idx)));

else
% Default split for task 1
    
    train_idx = [];
    test_idx = [];
    for i = 1:num_subjects
        inds = (i-1)*images_per_subject + (1:images_per_subject);
        perm = randperm(images_per_subject);
        train_count = floor(images_per_subject * train_ratio_task_1);
        train_idx = [train_idx; inds(perm(1:train_count))'];
        test_idx  = [test_idx;  inds(perm(train_count+1:end))'];
    end
end

% Apply to data
X_train = X(train_idx, :);
y_train = labels(train_idx);
X_test  = X(test_idx, :);
y_test  = labels(test_idx);


%% Step 4: MDA/PCA projection
if strcmp(proj_mode, 'mda')
    % MDA Projection
    W_mda = mda(X_train, y_train,m_mda);
    X_train_projected = X_train * W_mda;
    X_test_projected  = X_test  * W_mda;
elseif strcmp(proj_mode, 'pca')
    % PCA Projection
    W_pca = Pca(X_train, m_pca);
    X_train_projected = (X_train - mean(X_train, 1)) * W_pca;
    X_test_projected  = (X_test  - mean(X_train, 1)) * W_pca;  % note: use train mean
else
    fprintf('Wrong projection mode, select mda or pca.')
end

%% Step 5: Classification
y_pred = zeros(size(y_test));

% Bayes' Classifier Implementation
if strcmp(classifier, 'bayes')
    % Estimate class-wise Gaussian params
    classes = unique(y_train);
    C = length(classes);
    D_mda = size(X_train_projected, 2);
    class_means = zeros(C, D_mda);
    class_covs = zeros(D_mda, D_mda, C);
    for c = 1:C
        Xc = X_train_projected(y_train == c, :);
        class_means(c, :) = mean(Xc, 1);
        class_covs(:, :, c) = cov(Xc, 1) + 1e-1 * eye(D_mda);
    end
    % Apply Classifier
    for i = 1:length(y_test)
        x = X_test_projected(i, :)';
        scores = zeros(C, 1);
        for c = 1:C
            mu = class_means(c, :)';
            Sigma = class_covs(:, :, c);
            diff = x - mu;
            scores(c) = -0.5 * diff' * (Sigma \ diff) - 0.5 * log(det(Sigma));
        end
        [~, y_pred(i)] = max(scores);
    end


% K-NN Classifer Implementation
elseif strcmp(classifier, 'knn')
    for i = 1:length(y_test)
        test_point = X_test_projected(i, :);
        % Compute Euclidean distance to all training points
        distances = sum((X_train_projected - test_point).^2, 2);
        [~, sorted_idx] = sort(distances);
        nearest_labels = y_train(sorted_idx(1:k));
        % Assign the most frequent label
        y_pred(i) = mode(nearest_labels);
    end

% Kernel SVM Implementation
elseif strcmp(classifier, 'svm') 
    % normalize = @(X) X ./ sqrt(sum(X.^2, 2) + 1e-10);  % row-wise L2 norm
    % X_train_projected = normalize(X_train_projected);
    % X_test_projected = normalize(X_test_projected);

    % Selecting candidate parameters values for Cross Validation
    % For RBF:
    pairwise_dists = pdist2(X_train_projected, X_train_projected).^2;
    sigma2_median = median(pairwise_dists(:));
    sigma2_grid = sigma2_median * [0.25, 0.5, 1, 2, 4,10,100];
    % For Poly
    r_grid = [0,0.05,0.1,0.5,1,2, 3, 4, 5,6,7,8];


    % Set up K-fold cross-validation
    kfold = 5;
    indices = crossvalind('Kfold', y_train, kfold);
    best_acc = 0;
    % Cross-Validation Loop
    for s = 1:length(sigma2_grid) * strcmp(kernel_type, 'rbf') + ...
            length(r_grid) * strcmp(kernel_type, 'poly')

        if strcmp(kernel_type, 'rbf')
            sigma2 = sigma2_grid(s);
            K = @(x, y) exp(-pdist2(x, y, 'euclidean').^2 / sigma2);
        else
            r = r_grid(s);
            K = @(x, y) (x * y' + 1).^r;
        end

        acc_cv = 0;
        for k = 1:kfold
            val_idx = (indices == k);
            train_idx_cv = ~val_idx;

            Xtr = X_train_projected(train_idx_cv, :);
            ytr = 2 * (y_train(train_idx_cv) == 2) - 1;  % Convert to ±1
            Xval = X_train_projected(val_idx, :);
            yval = y_train(val_idx);

            % Compute kernel matrix
            Ktrain = K(Xtr, Xtr);
            N = size(Ktrain, 1);

            H = (ytr * ytr') .* Ktrain;
            f = -ones(N, 1);
            Aeq = ytr';
            beq = 0;
            lb = zeros(N, 1);
            C = 10;  % Regularization
            ub = C * ones(N, 1);

            options = optimoptions('quadprog', 'Display', 'off');
            alpha = quadprog(H + 1e-6*eye(N), f, [], [], Aeq, beq, lb, ub, [], options);

            % Support vectors
            sv = find((alpha > 1e-4) & (alpha < C));
            Xsv = Xtr(sv, :);
            ysv = ytr(sv);
            alpha_sv = alpha(sv);

            % Bias term (use mean of all support vectors)
            b = mean(ysv - sum((alpha_sv .* ysv) .* K(Xsv, Xsv), 1)');

            % Predict on validation
            scores = sum((alpha_sv .* ysv)' .* K(Xval, Xsv), 2) + b;
            ypred = (scores > 0) + 1;
            acc_cv = acc_cv + mean(ypred == yval);
        end

        acc_cv = acc_cv / kfold;

        if acc_cv > best_acc
            best_acc = acc_cv;
            best_kernel = K;
            if strcmp(kernel_type, 'rbf')
                best_param = sigma2;
            else
                best_param = r;
            end
        end
    end
    % fprintf('Best CV accuracy: %.2f%% (param = %.2f)\n', 100*best_acc, best_param);

    
    % Re-train on full train set using best param
    ytr = 2 * (y_train == 2) - 1;
    Ktrain = best_kernel(X_train_projected, X_train_projected);
    N = size(Ktrain, 1);
    % Optimization terms
    H = (ytr * ytr') .* Ktrain;
    f = -ones(N, 1);
    Aeq = ytr';
    beq = 0;
    lb = zeros(N, 1);
    C = 10;
    ub = C * ones(N, 1);
    alpha = quadprog(H + 1e-6*eye(N), f, [], [], Aeq, beq, lb, ub, [], options);
    sv = find(alpha > 1e-4);
    Xsv = X_train_projected(sv, :);
    ysv = ytr(sv);
    alpha_sv = alpha(sv);
    b = mean(ysv - sum((alpha_sv .* ysv)' .* best_kernel(Xsv, Xsv), 2));

    % Predict
    scores = sum((alpha_sv .* ysv)' .* best_kernel(X_test_projected, Xsv), 2) + b;
    y_pred = (scores > 0) + 1;


% Boosted-SVM Implementation
elseif strcmp(classifier, 'boosted_svm')
    % normalize = @(X) X ./ sqrt(sum(X.^2, 2) + 1e-10);  % row-wise L2 norm
    % X_train_projected = normalize(X_train_projected);
    % X_test_projected = normalize(X_test_projected);

    y_train = 2 * (y_train == 2) - 1;
    y_test = 2 * (y_test == 2) - 1;
    [alpha_all, models] = adaboost_linear_svm(X_train_projected, y_train, T);
    y_pred = adaboost_predict(X_test_projected, models, alpha_all);

end


%% Step 7: Display Accuracy
accuracy = mean(y_pred == y_test) * 100;
fprintf('Dataset: %s.mat | Task: %d | %s | Classifier: %s → Accuracy: %.2f%%\n', ...
    dataset_name,task, proj_mode,classifier, accuracy);
