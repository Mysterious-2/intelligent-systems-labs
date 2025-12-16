clc; clear; close all;

%% ==========================================
%  ADDITIONAL TASK: SURFACE APPROXIMATION
%  2 inputs (x1, x2) -> 1 output d(x1,x2)
%  MLP: 2–4–1 (tanh hidden, linear output)
%  Notation: w(dest, source)_layer
% ==========================================

%% --------- CREATE 2D TRAINING DATA ---------
x1 = linspace(0,1,100);
x2 = linspace(0,1,100);
[X1, X2] = meshgrid(x1, x2);

% Target surface (example nonlinear function)
D = (1 + 0.6*sin(2*pi*X1) + 0.3*cos(2*pi*X2)) / 2;

% Flatten grid -> vectors of samples
x1v = X1(:);
x2v = X2(:);
dv  = D(:);

N = length(dv);

%% --------- INITIALISE WEIGHTS ---------
rng('default');

% Layer 1 (hidden: 4 neurons, 2 inputs)
% w(dest, source)_layer: dest=hidden neuron index (1..4), source=input index (1..2)

% weights from input 1 (x1)
w11_1 = rand;  w21_1 = rand;  w31_1 = rand;  w41_1 = rand;
% weights from input 2 (x2)
w12_1 = rand;  w22_1 = rand;  w32_1 = rand;  w42_1 = rand;

% biases of hidden neurons
b1_1 = rand; b2_1 = rand; b3_1 = rand; b4_1 = rand;

% Layer 2 (output: 1 neuron, 4 inputs from hidden layer)
% dest = output neuron 1, source = hidden neuron index (1..4)
w11_2 = rand;
w12_2 = rand;
w13_2 = rand;
w14_2 = rand;

b1_2  = rand;

eta    = 0.003;
epochs = 15000;

%% --------- TRAINING LOOP ---------
for epoch = 1:epochs
    
    % (optional) track loss
    % E_total = 0;
    
    for n = 1:N
        
        x1n = x1v(n);
        x2n = x2v(n);
        dn  = dv(n);
        
        %% -------- FORWARD PASS --------
        v1_1 = w11_1*x1n + w12_1*x2n + b1_1;
        v2_1 = w21_1*x1n + w22_1*x2n + b2_1;
        v3_1 = w31_1*x1n + w32_1*x2n + b3_1;
        v4_1 = w41_1*x1n + w42_1*x2n + b4_1;
        
        y1_1 = tanh(v1_1);
        y2_1 = tanh(v2_1);
        y3_1 = tanh(v3_1);
        y4_1 = tanh(v4_1);
        
        v1_2 = w11_2*y1_1 + w12_2*y2_1 + w13_2*y3_1 + w14_2*y4_1 + b1_2;
        y    = v1_2;   % linear output
        
        %% -------- ERROR --------
        e = y - dn;
        % E_total = E_total + 0.5*e^2;
        
        %% -------- BACKPROP --------
        delta1_2 = e;   % linear output
        
        delta1_1 = (1 - y1_1^2) * w11_2 * delta1_2;
        delta2_1 = (1 - y2_1^2) * w12_2 * delta1_2;
        delta3_1 = (1 - y3_1^2) * w13_2 * delta1_2;
        delta4_1 = (1 - y4_1^2) * w14_2 * delta1_2;
        
        %% -------- WEIGHT UPDATES --------
        % Output layer
        w11_2 = w11_2 - eta*delta1_2*y1_1;
        w12_2 = w12_2 - eta*delta1_2*y2_1;
        w13_2 = w13_2 - eta*delta1_2*y3_1;
        w14_2 = w14_2 - eta*delta1_2*y4_1;
        b1_2  = b1_2  - eta*delta1_2;
        
        % Hidden layer neuron 1
        w11_1 = w11_1 - eta*delta1_1*x1n;
        w12_1 = w12_1 - eta*delta1_1*x2n;
        b1_1  = b1_1  - eta*delta1_1;
        
        % Hidden layer neuron 2
        w21_1 = w21_1 - eta*delta2_1*x1n;
        w22_1 = w22_1 - eta*delta2_1*x2n;
        b2_1  = b2_1  - eta*delta2_1;
        
        % Hidden layer neuron 3
        w31_1 = w31_1 - eta*delta3_1*x1n;
        w32_1 = w32_1 - eta*delta3_1*x2n;
        b3_1  = b3_1  - eta*delta3_1;
        
        % Hidden layer neuron 4
        w41_1 = w41_1 - eta*delta4_1*x1n;
        w42_1 = w42_1 - eta*delta4_1*x2n;
        b4_1  = b4_1  - eta*delta4_1;
    end
    
    % if mod(epoch,500)==0
    %     fprintf('Epoch %d done\n', epoch);
    % end
end

%% --------- TEST ON A DENSER GRID ---------
x1t = linspace(0, 1, 40);
x2t = linspace(0, 1, 40);
[X1t, X2t] = meshgrid(x1t, x2t);

% True target surface on test grid
D_test = (1 + 0.6*sin(2*pi*X1t) + 0.3*cos(2*pi*X2t)) / 2;

% MLP prediction on test grid
Y_pred = zeros(size(X1t));

for i = 1:size(X1t,1)
    for j = 1:size(X1t,2)
        x1n = X1t(i,j);
        x2n = X2t(i,j);
        
        v1_1 = w11_1*x1n + w12_1*x2n + b1_1;
        v2_1 = w21_1*x1n + w22_1*x2n + b2_1;
        v3_1 = w31_1*x1n + w32_1*x2n + b3_1;
        v4_1 = w41_1*x1n + w42_1*x2n + b4_1;
        
        y1_1 = tanh(v1_1);
        y2_1 = tanh(v2_1);
        y3_1 = tanh(v3_1);
        y4_1 = tanh(v4_1);
        
        Y_pred(i,j) = w11_2*y1_1 + w12_2*y2_1 + w13_2*y3_1 + w14_2*y4_1 + b1_2;
    end
end

%% --------- VISUALIZATION ---------

% TARGET surface
figure;
surf(X1t, X2t, D_test);
shading interp;
colormap parula;
colorbar;
title('Target Surface D(x1, x2)');
xlabel('x1'); ylabel('x2'); zlabel('Target');
view(45,30);

% MLP approximation surface
figure;
surf(X1t, X2t, Y_pred);
shading interp;
colormap parula;
colorbar;
title('MLP Approximated Surface');
xlabel('x1'); ylabel('x2'); zlabel('MLP Output');
view(45,30);

% Error surface
figure;
surf(X1t, X2t, Y_pred - D_test);
shading interp;
colormap hot;
colorbar;
title('Error Surface (MLP - Target)');
xlabel('x1'); ylabel('x2'); zlabel('Error');
view(45,30);

%% --------- COMBINED MESH PLOT (TARGET vs MLP) ---------

figure;

% Target surface as mesh (black)
mesh(X1t, X2t, D_test, 'EdgeColor', 'k', 'LineWidth', 1.2);
hold on;

% MLP surface as mesh (blue)
mesh(X1t, X2t, Y_pred, 'EdgeColor', 'b', 'LineWidth', 1.2);

title('Target Surface vs MLP Approximation (Mesh Comparison)');
xlabel('x1');
ylabel('x2');
zlabel('output');

legend('Target D(x1,x2)', 'MLP Y(x1,x2)');
grid on;
view(45,30);


