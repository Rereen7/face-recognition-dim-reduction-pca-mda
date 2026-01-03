function [W] = mda(X, y,m)
    classes = unique(y);
    C = length(classes);
    [N, D] = size(X);

    % Compute global mean
    mu = mean(X, 1);

    % Initialize scatter matrices
    Sw = zeros(D, D);  % within-class scatter
    Sb = zeros(D, D);  % between-class scatter

    for c = 1:C
        idx = (y == classes(c));
        Xc = X(idx, :);
        Nc = size(Xc, 1);
        muc = mean(Xc, 1);

        % Within-class scatter
        Sw = Sw + cov(Xc, 1) * Nc;

        % Between-class scatter
        d = (muc - mu)';
        Sb = Sb + Nc * (d * d');
        Sw = Sw + 1e-10 * eye(size(Sw));



    end

    % Solve generalized eigenvalue problem Sb*w = lambda*Sw*w
    % [V, ~] = eigs(Sb, Sw, C/2);  % max C-1 discriminant components
    [V_all, D] = eig(Sb, Sw);  % Generalized eigenvalue problem
    [~, idx] = sort(diag(D), 'descend');  % Sort by eigenvalue magnitude
    V = V_all(:, idx(1:m));  % Take top C-1 directions
    W = V;  % Projection matrix
end

