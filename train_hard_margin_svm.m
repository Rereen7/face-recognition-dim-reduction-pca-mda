% function model = train_hard_margin_svm(X, y)
% % Train a hard-margin linear SVM using quadprog (primal form)
% % Inputs:
% %   X: N x D feature matrix
% %   y: N x 1 label vector (-1/+1)
% % Outputs:
% %   model.w: weight vector
% %   model.b: bias term
% 
% [N, D] = size(X);
% 
% % Variables: [w; b] -> size = D+1
% H = blkdiag(eye(D), 0);  % Only penalize w, not b
% f = zeros(D+1, 1);       % No linear cost
% 
% % Constraints: y_i * (w^T x_i + b) >= 1  ⇒  -y_i*(x_i*w + b) <= -1
% A = -[diag(y) * X, y];
% b_vec = -ones(N, 1);
% 
% % Solve QP
% options = optimoptions('quadprog', 'Display', 'off');
% sol = quadprog(H, f, A, b_vec, [], [], [], [], [], options);
% 
% % Extract solution
% w_opt = sol(1:D);
% b_opt = sol(D+1);
% 
% model.w = w_opt;
% model.b = b_opt;
% end
function model = train_hard_margin_svm(X, y)
% Train a linear SVM using stochastic gradient descent (SGD)
% This approximates a hard-margin behavior for boosting
% Inputs:
%   X: N x D matrix of features
%   y: N x 1 labels in {-1, +1}
% Output:
%   model struct with fields: w and b

[N, D] = size(X);
% w = zeros(D, 1);
% b = 0;
w = randn(D, 1);
b = randn(1);
% Hyperparameters
epochs = 10;
lr = 0.001;           % learning rate
lambda = 1e-5;       % small regularization to prevent overfitting

% SGD training loop
for epoch = 1:epochs
    idx = randperm(N);  % shuffle samples each epoch
    for i = idx
        xi = X(i, :)';
        yi = y(i);

        % Hinge loss condition
        if yi * (w' * xi + b) < 1
            % Misclassified or on margin: apply update
            w = (1 - lr * lambda) * w + lr * yi * xi;
            b = b + lr * yi;
        else
            % Correct and outside margin: just regularize
            w = (1 - lr * lambda) * w;
            % b remains unchanged
        end
    end
end

% Return trained model
model.w = w;
model.b = b;
end
