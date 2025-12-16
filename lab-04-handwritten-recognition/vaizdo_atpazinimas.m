close all; clear all; clc;

%% =========================
%  TRAINING STAGE
% =========================

pavadinimas = 'train_digits.png';      % <-- your training image
train_rows = 5;                       % <-- IMPORTANT: 6 rows
numClasses = 6;                       % digits 1..6

pozymiai_tinklo_mokymui = pozymiai_raidems_atpazinti(pavadinimas, train_rows);

% Features matrix (each column = one digit sample)
P = cell2mat(pozymiai_tinklo_mokymui);

% Targets: each row is [1 2 3 4 5] => eye(5)
T = [eye(numClasses), eye(numClasses), eye(numClasses), eye(numClasses), eye(numClasses)];

% Train RBF network
num_neurons = 9;     
tinklas = newrb(P, T, 0, 1, num_neurons);

%% =========================
%  VALIDATION USING 2nd ROW OF TRAINING IMAGE
% =========================

% If each row has 5 symbols:
% row1 = columns 1..5
% row2 = columns 6..10
P2 = P(:, 7:12);

Y2 = sim(tinklas, P2);
[~, b2] = max(Y2);

% Convert classes -> digits
raidziu_sk = size(P2,2);
atsakymas = [];

for k = 1:raidziu_sk
    switch b2(k)
        case 1, atsakymas = [atsakymas, '1'];
        case 2, atsakymas = [atsakymas, '2'];
        case 3, atsakymas = [atsakymas, '3'];
        case 4, atsakymas = [atsakymas, '4'];
        case 5, atsakymas = [atsakymas, '5'];
        case 6, atsakymas = [atsakymas, '6'];
    end
end

disp(atsakymas);
figure(7), text(0.1,0.5,atsakymas,'FontSize',38), axis off;

%% =========================
%  TESTING ON EXTERNAL IMAGE
% =========================

pavadinimas = 'test_data1.png';   % <-- your test image (1 line)
pozymiai_patikrai = pozymiai_raidems_atpazinti(pavadinimas, 1);

P2 = cell2mat(pozymiai_patikrai);
Y2 = sim(tinklas, P2);
[~, b2] = max(Y2);

raidziu_sk = size(P2,2);
atsakymas = [];

for k = 1:raidziu_sk
    switch b2(k)
        case 1, atsakymas = [atsakymas, '1'];
        case 2, atsakymas = [atsakymas, '2'];
        case 3, atsakymas = [atsakymas, '3'];
        case 4, atsakymas = [atsakymas, '4'];
        case 5, atsakymas = [atsakymas, '5'];
        case 6, atsakymas = [atsakymas, '6'];
    end
end

disp(atsakymas);
figure(8), text(0.1,0.5,atsakymas,'FontSize',38), axis off;
