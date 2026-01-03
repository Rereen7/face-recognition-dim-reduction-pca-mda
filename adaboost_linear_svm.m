function [alpha_all, models] = adaboost_linear_svm(X, y, T)
% AdaBoost with Linear SVMs from scratch using quadprog
% Inputs:
%   X: N x D input matrix (features)
%   y: N x 1 labels (-1 or +1)
%   T: number of boosting rounds


N = size(X, 1);
D = size(X, 2);
w = ones(N, 1) / N;  % initial weights

models = cell(T, 1);     % store each SVM model
alpha_all = zeros(T, 1); % classifier weights
t = 1;
% for t = 1:T
while (t <= T)

    model = train_hard_margin_svm(X,y);
    y_pred = sign(X* model.w + model.b);

    % Compute weighted error
    err = sum(w .* (y_pred ~= y));
    

    err = max(min(err, 0.5), 1e-10);  % clip error between [1e-10, 1 - 1e-10]

    % Compute classifier weight
    alpha_t = 0.5 * log((1 - err) / err);
    if alpha_t == 0
        continue
    end
    % Update sample weights
    w = w .* exp(-alpha_t * y .* y_pred);
    w = w / sum(w);  % normalize

    % Store results
    alpha_all(t) = alpha_t;
    models{t} = model;

    fprintf('Round %d: err = %.4f, alpha = %.4f\n', t, err, alpha_t);
    
    t = t+1;
    
end
end
