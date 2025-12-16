clc; clear;

%% CREATE DATA (20 points)
x = 0.1 : 1/22 : 1;

% target function: (1 + 0.6 sin(2πx/0.7) + 0.3 sin(2πx)) / 2
d = (1 + 0.6 * sin(2 * pi * x / 0.7) + 0.3 * sin(2 * pi * x)) / 2;

%% INITIALISE WEIGHTS

% Layer 1 (hidden, 4 neurons, 1 input)
% w(dest, source)_layer   -> dest = hidden neuron index, source = input index

w11_1 = rand(1);   % to hidden neuron 1 from input 1
w21_1 = rand(1);   % to hidden neuron 2 from input 1
w31_1 = rand(1);   % to hidden neuron 3 from input 1
w41_1 = rand(1);   % to hidden neuron 4 from input 1

b1_1  = rand(1);   % bias for hidden neuron 1
b2_1  = rand(1);   % bias for hidden neuron 2
b3_1  = rand(1);   % bias for hidden neuron 3
b4_1  = rand(1);   % bias for hidden neuron 4

% Layer 2 (output, 1 neuron, 4 inputs from hidden layer)
% w(dest, source)_layer -> dest = output neuron 1, source = hidden neuron index

w11_2 = rand(1);   % to output neuron 1 from hidden neuron 1
w12_2 = rand(1);   % to output neuron 1 from hidden neuron 2
w13_2 = rand(1);   % to output neuron 1 from hidden neuron 3
w14_2 = rand(1);   % to output neuron 1 from hidden neuron 4

b1_2  = rand(1);   % bias for output neuron 1

eta    = 0.01;       % learning rate
epochs = 10000000;     % number of passes over data

%% TRAINING LOOP
N = length(x);

for epoch = 1:epochs
    
    E_total = 0;   % total error this epoch
    
    for n = 1:N
        
        xn = x(n);     % input
        dn = d(n);     % desired output
        
        %%  FORWARD PASS 
        % Layer 1 (hidden, tanh)
        v1_1 = w11_1 * xn + b1_1;    % net for hidden neuron 1
        v2_1 = w21_1 * xn + b2_1;    % net for hidden neuron 2
        v3_1 = w31_1 * xn + b3_1;    % net for hidden neuron 3
        v4_1 = w41_1 * xn + b4_1;    % net for hidden neuron 4
        
        y1_1 = tanh(v1_1);           % output of hidden neuron 1
        y2_1 = tanh(v2_1);           % output of hidden neuron 2
        y3_1 = tanh(v3_1);           % output of hidden neuron 3
        y4_1 = tanh(v4_1);           % output of hidden neuron 4
        
        % Layer 2 (output, linear)
        v1_2 = w11_2 * y1_1 + w12_2 * y2_1 + w13_2 * y3_1 + w14_2 * y4_1 + b1_2;   % net at output neuron 1
        y1_2 = v1_2;                 % linear activation
        
        %%  ERROR 
        e = y1_2 - dn;                    % error for sample n
        E_total = E_total + 0.5 * e^2;    % accumulate 1/2 e^2
        
        %% BACKPROP
        % Output layer delta (linear)
        delta1_2 = e;   % dE/dv1_2
        
        % Hidden layer deltas (tanh: derivative = 1 - y^2)
        delta1_1 = (1 - y1_1^2) * w11_2 * delta1_2;  % for hidden neuron 1
        delta2_1 = (1 - y2_1^2) * w12_2 * delta1_2;  % for hidden neuron 2
        delta3_1 = (1 - y3_1^2) * w13_2 * delta1_2;  % for hidden neuron 3
        delta4_1 = (1 - y4_1^2) * w14_2 * delta1_2;  % for hidden neuron 4
        
        %%  WEIGHT UPDATES 
        % Layer 2 (output)
        w11_2 = w11_2 - eta * delta1_2 * y1_1;
        w12_2 = w12_2 - eta * delta1_2 * y2_1;
        w13_2 = w13_2 - eta * delta1_2 * y3_1;
        w14_2 = w14_2 - eta * delta1_2 * y4_1;
        b1_2  = b1_2  - eta * delta1_2;
        
        % Layer 1 (hidden)
        w11_1 = w11_1 - eta * delta1_1 * xn;
        w21_1 = w21_1 - eta * delta2_1 * xn;
        w31_1 = w31_1 - eta * delta3_1 * xn;
        w41_1 = w41_1 - eta * delta4_1 * xn;
        
        b1_1  = b1_1  - eta * delta1_1;
        b2_1  = b2_1  - eta * delta2_1;
        b3_1  = b3_1  - eta * delta3_1;
        b4_1  = b4_1  - eta * delta4_1;
    end
end

%% TEST / PLOT FINAL APPROXIMATION
x_test = linspace(0, 1, 200);   % new points
Ntest = length(x_test);

y_test = zeros(size(x_test));
for n = 1:Ntest
    xn = x_test(n);

    v1_1 = w11_1 * xn + b1_1;
    v2_1 = w21_1 * xn + b2_1;
    v3_1 = w31_1 * xn + b3_1;
    v4_1 = w41_1 * xn + b4_1;

    y1_1 = tanh(v1_1);
    y2_1 = tanh(v2_1);
    y3_1 = tanh(v3_1);
    y4_1 = tanh(v4_1);

    v1_2 = w11_2*y1_1 + w12_2*y2_1 + w13_2*y3_1 + w14_2*y4_1 + b1_2;

    y_test(n) = v1_2;
end

%% TEST / PLOT FINAL APPROXIMATION
x_test = linspace(0, 1, 200);   % many new points
Ntest = length(x_test);

% Compute TRUE FUNCTION on x_test
d_test = (1 + 0.6 * sin(2 * pi * x_test / 0.7) + 0.3 * sin(2 * pi * x_test)) / 2;

% Compute MLP output on x_test
y_test = zeros(size(x_test));

for n = 1:Ntest
    xn = x_test(n);

    v1_1 = w11_1 * xn + b1_1;
    v2_1 = w21_1 * xn + b2_1;
    v3_1 = w31_1 * xn + b3_1;
    v4_1 = w41_1 * xn + b4_1;

    y1_1 = tanh(v1_1);
    y2_1 = tanh(v2_1);
    y3_1 = tanh(v3_1);
    y4_1 = tanh(v4_1);

    v1_2 = w11_2*y1_1 + w12_2*y2_1 + w13_2*y3_1 + w14_2*y4_1 + b1_2;
    y_test(n) = v1_2;
end

figure;

% 1) ORIGINAL FUNCTION (smooth red line)
plot(x_test, d_test, 'r-', 'LineWidth', 2);   
hold on;

% 2) TRAINING DATA (red points)
plot(x, d, 'ro', 'MarkerSize', 6, 'LineWidth', 1.2);  

% 3) MLP OUTPUT (blue line)
plot(x_test, y_test, 'b-', 'LineWidth', 2);   

legend('Original function d(x)', 'Training samples', 'MLP approximation y(x)');
xlabel('x');
ylabel('output');
title('1–4–1 MLP Function Approximation');
grid on;


