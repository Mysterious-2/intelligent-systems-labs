clc
% Classification using perceptron

% Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

% Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

% Put apple and pear images into cell arrays
apple_imgs = {A1, A2, A3, A4, A5, A6, A7, A8, A9};
pear_imgs  = {P1, P2, P3, P4};

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

% selecting features (colour, roundness) – 3 apples and 2 pears: A1, A2, A3, P1, P2
x1 = [hsv_A(1) hsv_A(2) hsv_A(3) hsv_P(1) hsv_P(2)];
x2 = [met_A(1) met_A(2) met_A(3) met_P(1) met_P(2)];

% estimated features in matrix P (2 x 5):
P  = [x1; x2];

% Desired target vector: apples = 1, pears = -1
T = [1; 1; 1; -1; -1];

%% train single perceptron with two inputs and one output

% generate random initial values of w1, w2 and b
w1 = randn(1);
w2 = randn(1);
b  = randn(1);

% Initial forward pass
v = zeros(1,5);
y = zeros(1,5);
e = zeros(1,5);
for n = 1:5
    v(n) = P(1,n)*w1 + P(2,n)*w2 + b;   % weighted sum
    
    if v(n) > 0
        y(n) = 1;
    else
        y(n) = -1;
    end
    
    e(n) = T(n) - y(n);                 % error
end
E = sum(abs(e));   
fprintf('Initial total error before training: %d\n', E);

% =======================
% Perceptron training loop
% =======================
eta = 0.1;   % learning rate

% initialise total error so that the loop starts
e_total = 1;

while e_total ~= 0  
    
    % 1) TRAIN: go through all 5 examples and update weights 
    for n = 1:5
        x1n = P(1,n);          % first feature (colour)
        x2n = P(2,n);          % second feature (roundness)
        
        % calculate output for current example
        v_n = x1n*w1 + x2n*w2 + b;
        if v_n > 0
            y_n = 1;
        else
            y_n = -1;
        end
        
        % calculate error for current example
        en = T(n) - y_n;
        
        % update parameters using current inputs and current error
        w1 = w1 + eta*en*x1n;
        w2 = w2 + eta*en*x2n;
        b  = b  + eta*en;
    end
    
    % 2) TEST: compute total error on all 5 examples with new weights 
    e_total = 0;   % reset total error
    
    for n = 1:5
        v_n = P(1,n)*w1 + P(2,n)*w2 + b;
        if v_n > 0
            y_n = 1;
        else
            y_n = -1;
        end
        
        en     = T(n) - y_n;
        e_total = e_total + abs(en);   % accumulate absolute error
    end
end

% =======================
% FINAL RESULTS
% =======================

% Final outputs for training examples
final_y = zeros(1,5);
for n = 1:5
    v_n = P(1,n)*w1 + P(2,n)*w2 + b;
    if v_n > 0
        final_y(n) = 1;
    else
        final_y(n) = -1;
    end
end

% Print final weights and bias
disp('Final weights and bias after training:');
fprintf('w1 = %.4f, w2 = %.4f, b = %.4f\n', w1, w2, b);

% Print targets vs outputs
disp('Targets (T) and perceptron outputs (final_y) for 5 training examples:');
disp('T:');
disp(T');          % transpose to show as row
disp('final_y:');
disp(final_y);


%% TESTING ON UNUSED IMAGES (A4–A9, P3, P4)

test_imgs   = {A4, A5, A6, A7, A8, A9, P3, P4};
test_labels = [1 1 1 1 1 1 -1 -1];   % apples = 1, pears = -1

num_test    = length(test_imgs);
test_y      = zeros(1, num_test);

for k = 1:num_test
    % extract features for test image
    hsv_k = spalva_color(test_imgs{k});
    met_k = apvalumas_roundness(test_imgs{k});
    
    % classify using trained perceptron 
    v_k = hsv_k*w1 + met_k*w2 + b;
    if v_k > 0
        test_y(k) = 1;
    else
        test_y(k) = -1;
    end
end

disp('True labels (unused images):');
disp(test_labels);

disp('Predicted labels (unused images):');
disp(test_y);

accuracy = sum(test_y == test_labels) / num_test;
fprintf('Accuracy on unused images: %.2f\n', accuracy);
