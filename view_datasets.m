% Load all datasets
load('data.mat');         % loads variable: face (24x21x600)
load('pose.mat');         % loads variable: pose (48x40x13x68)
load('illumination.mat'); % loads variable: illum (1920x21x68)

%% View samples from data.mat
figure('Name', 'Data.mat – 3 face conditions');
subject_id = randi(200);
for i = 1:3
    subplot(1, 3, i);
    img = face(:, :, 3*subject_id - 3 + i);
    imshow(img, []);
    title(["Face " + i]);
end

%% View samples from pose.mat
figure('Name', 'Pose.mat – 13 poses');
subject_id = randi(68);
for i = 1:13
    subplot(2, 7, i);
    img = pose(:, :, i, subject_id);
    imshow(img, []);
    title("Pose " + i);
end

%% View samples from illumination.mat
figure('Name', 'Illumination.mat – 21 lighting conditions');
subject_id = randi(68);
for i = 1:21
    subplot(3, 7, i);
    img = reshape(illum(:, i, subject_id), 48, 40);
    imshow(img, []);
    title("Light " + i);
end
