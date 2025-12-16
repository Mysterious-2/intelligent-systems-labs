clc; clear; close all;

%% CREATE DATA (20 points)
x = 0.1 : 1/22 : 1;   % 20 training samples

% target function: (1 + 0.6 sin(2πx/0.7) + 0.3 sin(2πx)) / 2
d = (1 + 0.6 * sin(2 * pi * x / 0.7) + 0.3 * sin(2 * pi * x)) / 2;

%% RBF PARAMETERS (chosen manually)
% Two Gaussians: phi = exp(-(x-c)^2 / (2*r^2))

c1 = 0.2;    % center of first RBF
r1 = 0.14;    % width  of first RBF

c2 = 0.9;    % center of second RBF
r2 = 0.14;    % width  of second RBF

%% INITIALISE OUTPUT LAYER PARAMETERS (2 weights + bias)
w1 = randn(1);
w2 = randn(1);
b  = randn(1);

eta    = 0.05;       % learning rate
epochs = 1000000;      % number of passes over data

N = length(x);

%% TRAINING LOOP (perceptron-style for 2nd layer)
for epoch = 1:epochs
    
    E_total = 0;   % total error this epoch
    
    for n = 1:N
        
        xn = x(n);     % input
        dn = d(n);     % desired output
        
        %  FORWARD PASS 
        % Hidden layer: two fixed RBFs
        phi1 = exp( -(xn - c1)^2 / (2 * r1^2) );
        phi2 = exp( -(xn - c2)^2 / (2 * r2^2) );
        
        % Output layer: linear neuron
        y = w1 * phi1 + w2 * phi2 + b;
        
        %  ERROR 
        e = y - dn;                 % error for sample n
        E_total = E_total + 0.5 * e^2;
        
        %  PARAMETER UPDATE (2nd layer only) 
        % Gradient descent on 1/2 * e^2
        w1 = w1 - eta * e * phi1;
        w2 = w2 - eta * e * phi2;
        b  = b  - eta * e;
    end
    
    % Uncomment this if you want to see convergence:
    % if mod(epoch, 1000) == 0
    %     fprintf('Epoch %d, E_total = %.6f\n', epoch, E_total);
    % end
    
    % Early stopping (optional)
    % if E_total < 1e-5
    %     break;
    % end
end

%% TEST / PLOT FINAL APPROXIMATION

x_test = linspace(0, 1, 200);
Ntest = length(x_test);

% true function on dense grid
d_test = (1 + 0.6 * sin(2 * pi * x_test / 0.7) + 0.3 * sin(2 * pi * x_test)) / 2;

% RBF network output on dense grid
y_test = zeros(size(x_test));

for n = 1:Ntest
    xn = x_test(n);
    
    phi1 = exp( -(xn - c1)^2 / (2 * r1^2) );
    phi2 = exp( -(xn - c2)^2 / (2 * r2^2) );
    
    y_test(n) = w1 * phi1 + w2 * phi2 + b;
end

figure;
plot(x_test, d_test, 'r-', 'LineWidth', 2);    % true function
hold on;
plot(x, d, 'ro', 'MarkerSize', 6, 'LineWidth', 1.2);  % training samples
plot(x_test, y_test, 'b-', 'LineWidth', 2);    % RBF approximation

legend('Original function d(x)', 'Training samples', 'RBF approximation y(x)');
xlabel('x');
ylabel('output');
title('1–2–1 RBF Network Function Approximation');
grid on;
