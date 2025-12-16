clc; clear; close all;

%% =========================
%  DATA (20 training samples)
%  x in [0.1, 1], step 1/22 -> 20 points
%  d(x) = (1 + 0.6 sin(2πx/0.7) + 0.3 sin(2πx)) / 2
%% =========================
x = 0.1 : 1/22 : 1;
d = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x)) / 2;
N = length(x);

%% =========================
%  INITIAL PARAMETERS
%  Two Gaussian RBFs:
%    φ_i(x) = exp( - (x - c_i)^2 / (2 r_i^2) )
%  Output:
%    y(x) = w1*φ1(x) + w2*φ2(x) + b
%% =========================
c1 = 0.25;  c2 = 0.75;
r1 = 0.18;  r2 = 0.18;

% learning rates for center/radius updates
eta_c = 0.005;
eta_r = 0.001;

epochs = 2000;

% stability clamps (keep parameters meaningful)
c_min = 0;    c_max = 1;
r_min = 0.05; r_max = 0.50;

%% =========================
%  TRAINING LOOP
%  We do:
%   1) compute φ1, φ2 using current c,r
%   2) compute best w1,w2,b via least squares (stable)
%   3) update c and r by gradient descent using chain rule
%% =========================
for epoch = 1:epochs

    % -------- 1) RBF FEATURES (hidden layer outputs) --------
    % φ1(x) = exp( - (x - c1)^2 / (2 r1^2) )
    % φ2(x) = exp( - (x - c2)^2 / (2 r2^2) )
    phi1 = exp( - (x - c1).^2 ./ (2*r1^2) );
    phi2 = exp( - (x - c2).^2 ./ (2*r2^2) );

    % -------- 2) OUTPUT LAYER (solve w1,w2,b exactly) --------
    % For all samples we want: d ≈ w1*phi1 + w2*phi2 + b
    %
    % Matrix form:
    %   A = [phi1  phi2  1]
    %   theta = [w1; w2; b]
    %   A*theta ≈ d
    %
    % Least squares solution:
    %   theta = argmin ||A*theta - d||^2
    A = [phi1(:), phi2(:), ones(N,1)];
    theta = A \ d(:);     % solves least squares
    w1 = theta(1); w2 = theta(2); b = theta(3);

    % predictions and error:
    y = (A*theta).';      % y_n
    e = y - d;            % e_n = y_n - d_n

    % Loss:
    %   E = 1/2 * sum_n e_n^2

    % -------- 3) "BACKPROP" PART: gradients for c and r --------
    % Chain rule:
    %   dE/dp = sum_n ( e_n * dy_n/dp )
    %
    % Since y_n = w1*φ1n + w2*φ2n + b:
    %   dy/dc1 = w1 * dφ1/dc1
    %   dy/dr1 = w1 * dφ1/dr1
    %   dy/dc2 = w2 * dφ2/dc2
    %   dy/dr2 = w2 * dφ2/dr2

    % Derivatives of Gaussian φ(x)=exp(-(x-c)^2/(2r^2)):
    %
    %   dφ/dc = φ * (x - c) / r^2
    %   dφ/dr = φ * (x - c)^2 / r^3

    dphi1_dc1 = phi1 .* (x - c1) / (r1^2);
    dphi2_dc2 = phi2 .* (x - c2) / (r2^2);

    dphi1_dr1 = phi1 .* (x - c1).^2 / (r1^3);
    dphi2_dr2 = phi2 .* (x - c2).^2 / (r2^3);

    % Gradients of E:
    %   dE/dc1 = sum_n [ e_n * (w1 * dφ1n/dc1) ]
    %   dE/dr1 = sum_n [ e_n * (w1 * dφ1n/dr1) ]
    grad_c1 = sum( e .* (w1 * dphi1_dc1) );
    grad_c2 = sum( e .* (w2 * dphi2_dc2) );

    grad_r1 = sum( e .* (w1 * dphi1_dr1) );
    grad_r2 = sum( e .* (w2 * dphi2_dr2) );

    % Gradient descent updates:
    %   p <- p - η * dE/dp
    c1 = c1 - eta_c * grad_c1;
    c2 = c2 - eta_c * grad_c2;

    r1 = r1 - eta_r * grad_r1;
    r2 = r2 - eta_r * grad_r2;

    % -------- 4) CLAMP FOR STABILITY --------
    % Keep centers in [0,1], radii in [r_min, r_max]
    c1 = min(max(c1, c_min), c_max);
    c2 = min(max(c2, c_min), c_max);

    r1 = min(max(r1, r_min), r_max);
    r2 = min(max(r2, r_min), r_max);
end

%% =========================
%  TEST / PLOT
%% =========================
x_test = linspace(0,1,200);
d_test = (1 + 0.6*sin(2*pi*x_test/0.7) + 0.3*sin(2*pi*x_test)) / 2;

phi1t = exp( - (x_test - c1).^2 ./ (2*r1^2) );
phi2t = exp( - (x_test - c2).^2 ./ (2*r2^2) );
y_test = w1*phi1t + w2*phi2t + b;

figure;
plot(x_test, d_test, 'r-', 'LineWidth', 2); hold on;
plot(x, d, 'ro', 'MarkerSize', 6, 'LineWidth', 1.2);
plot(x_test, y_test, 'b-', 'LineWidth', 2);
grid on;
legend('Original d(x)','Training samples','RBF y(x)');
title('Trainable RBF: update c,r (gradient) + solve w (LS)');

fprintf('Final: w1=%.4f, w2=%.4f, b=%.4f\n', w1,w2,b);
fprintf('Final: c1=%.4f, r1=%.4f, c2=%.4f, r2=%.4f\n', c1,r1,c2,r2);
