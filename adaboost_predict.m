function y_pred = adaboost_predict(X, models, alpha_all)
% Predict using boosted SVMs
% Inputs:
%   X: test samples (N x D)
%   models: cell array of SVM models
%   alpha_all: weights of each model

T = length(models);
N = size(X, 1);
votes = zeros(N, 1);

for t = 1:T
    y_t = sign(X * models{t}.w + models{t}.b);
    votes = votes + alpha_all(t) * y_t;
    
end

y_pred = sign(votes);
end
