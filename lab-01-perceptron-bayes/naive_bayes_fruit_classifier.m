clc
clear

% Classification using Naive Bayes classifier

% =======================
% Reading images
% =======================

% Reading apple images
A1 = imread('apple_04.jpg');
A2 = imread('apple_05.jpg');
A3 = imread('apple_06.jpg');
A4 = imread('apple_07.jpg');
A5 = imread('apple_11.jpg');
A6 = imread('apple_12.jpg');
A7 = imread('apple_13.jpg');
A8 = imread('apple_17.jpg');
A9 = imread('apple_19.jpg');

% Reading pear images
P1 = imread('pear_01.jpg');
P2 = imread('pear_02.jpg');
P3 = imread('pear_03.jpg');
P4 = imread('pear_09.jpg');

% Put apple and pear images into cell arrays
apple_imgs = {A1, A2, A3, A4, A5, A6, A7, A8, A9};
pear_imgs  = {P1, P2, P3, P4};

% =======================
% Feature extraction
% =======================

% Preallocate feature arrays
num_apples = length(apple_imgs);
num_pears  = length(pear_imgs);

hsv_A = zeros(1, num_apples);
met_A = zeros(1, num_apples);

hsv_P = zeros(1, num_pears);
met_P = zeros(1, num_pears);

% Compute features for apples
for i = 1:num_apples
    hsv_A(i) = spalva_color(apple_imgs{i});
    met_A(i) = apvalumas_roundness(apple_imgs{i});
end

% Compute features for pears
for i = 1:num_pears
    hsv_P(i) = spalva_color(pear_imgs{i});
    met_P(i) = apvalumas_roundness(pear_imgs{i});
end

% =======================
% Training data selection
% =======================

% Selecting features (colour, roundness) – A1, A2, A3, P1, P2
x1 = [hsv_A(1) hsv_A(2) hsv_A(3) hsv_P(1) hsv_P(2)];
x2 = [met_A(1) met_A(2) met_A(3) met_P(1) met_P(2)];

% Feature matrix (2 x 5)
P = [x1; x2];

% Desired target vector: apples = 1, pears = -1
T = [1; 1; 1; -1; -1];

train_X = P;
train_T = T;

% =======================
% Naive Bayes training
% =======================

% Split samples by class
idxApple = find(train_T == 1);
idxPear  = find(train_T == -1);

X_apple = train_X(:, idxApple);
X_pear  = train_X(:, idxPear);

% Prior probabilities
prior_apple = numel(idxApple) / numel(train_T);
prior_pear  = numel(idxPear)  / numel(train_T);

% Mean and variance estimation
mu_apple  = mean(X_apple, 2);
mu_pear   = mean(X_pear, 2);

var_apple = var(X_apple, 0, 2);
var_pear  = var(X_pear, 0, 2);

% Prevent zero variance
var_apple(var_apple == 0) = 1e-6;
var_pear(var_pear == 0)   = 1e-6;

% Gaussian probability density function
gauss_pdf = @(x, mu, v) 1 ./ sqrt(2*pi*v) .* exp(-(x - mu).^2 ./ (2*v));

% =======================
% Classification of training samples
% =======================

nb_pred_train = zeros(1, 5);

for n = 1:5
    x = train_X(:, n);
    
    % Likelihoods
    p_x_apple = gauss_pdf(x(1), mu_apple(1), var_apple(1)) * ...
                gauss_pdf(x(2), mu_apple(2), var_apple(2));
            
    p_x_pear  = gauss_pdf(x(1), mu_pear(1),  var_pear(1))  * ...
                gauss_pdf(x(2), mu_pear(2),  var_pear(2));
    
    % Posterior scores
    score_apple = prior_apple * p_x_apple;
    score_pear  = prior_pear  * p_x_pear;
    
    % Decision rule
    if score_apple > score_pear
        nb_pred_train(n) = 1;
    else
        nb_pred_train(n) = -1;
    end
end

disp('Naive Bayes – TRAINING samples (A1, A2, A3, P1, P2):');
disp('True labels:'); disp(train_T');
disp('Predicted labels:'); disp(nb_pred_train);

train_accuracy = sum(nb_pred_train == train_T') / numel(train_T);
fprintf('Training accuracy: %.2f\n', train_accuracy);

% =======================
% Testing on unused images
% =======================

test_x1 = [hsv_A(4:9) hsv_P(3:4)];
test_x2 = [met_A(4:9) met_P(3:4)];
test_X  = [test_x1; test_x2];

test_labels = [1 1 1 1 1 1 -1 -1];   % apples = 1, pears = -1
num_test = length(test_labels);

nb_pred_test = zeros(1, num_test);

for k = 1:num_test
    x = test_X(:, k);
    
    p_x_apple = gauss_pdf(x(1), mu_apple(1), var_apple(1)) * ...
                gauss_pdf(x(2), mu_apple(2), var_apple(2));
            
    p_x_pear  = gauss_pdf(x(1), mu_pear(1),  var_pear(1))  * ...
                gauss_pdf(x(2), mu_pear(2),  var_pear(2));
    
    score_apple = prior_apple * p_x_apple;
    score_pear  = prior_pear  * p_x_pear;
    
    if score_apple > score_pear
        nb_pred_test(k) = 1;
    else
        nb_pred_test(k) = -1;
    end
end

disp('Naive Bayes – UNUSED images (A4–A9, P3, P4):');
disp('True labels:'); disp(test_labels);
disp('Predicted labels:'); disp(nb_pred_test);

test_accuracy = sum(nb_pred_test == test_labels) / num_test;
fprintf('Accuracy on unused images: %.2f\n', test_accuracy);
