function W = Pca(X, m)
% PCA_FROM_SCRATCH Compute PCA projection matrix from data
% Inputs:
%   X : N x D data matrix (each row is a sample)
%   m : number of principal components to retain
% Output:
%   W : D x m matrix with top-m eigenvectors (each column is a component)

% Step 1: Center the data
X_centered = X - mean(X, 1);

% Step 2: Compute covariance matrix
C = cov(X_centered);  % D x D
C = C + 1e-1 * eye(size(C));
% Step 3: Eigen-decomposition
[Vecs, Vals] = eig(C);
[~, idx] = sort(diag(Vals), 'descend');  % sort eigenvalues descending
W = Vecs(:, idx(1:m));  % top-m eigenvectors (D x m)

end
