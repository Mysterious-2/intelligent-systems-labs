close all; clear all; clc;

%% ================================================================
%  TRAINING STAGE (MLP)
% ================================================================

pavadinimas = 'train_digits.png';
train_rows = 5;      % <-- 5 rows in training image
numClasses = 6;      % <-- digits 1..6

pozymiai_tinklo_mokymui = pozymiai_raidems_atpazinti(pavadinimas, train_rows);

% Features matrix: each column = one symbol feature vector
P = cell2mat(pozymiai_tinklo_mokymui);

% Normalize features (IMPORTANT for MLP stability)
[Pn, ps] = mapminmax(P, 0, 1);

% Targets: each row is 1 2 3 4 5 6 -> eye(6), repeated 5 times
T = repmat(eye(numClasses), 1, train_rows);

%% ================================================================
%  ADDITIONAL TASK: MULTILAYER PERCEPTRON (MLP)
% ================================================================

hidden_layer_sizes = [20 20];                 % two hidden layers
mlp_net = feedforwardnet(hidden_layer_sizes, 'trainlm');

% Use all data for training (small dataset, lab-style)
mlp_net.divideFcn = 'dividetrain';

mlp_net.trainParam.epochs = 500;
mlp_net.trainParam.goal   = 1e-4;

mlp_net = train(mlp_net, Pn, T);

%% ================================================================
%  VALIDATION USING PART OF TRAINING DATA (2nd row)
% ================================================================

% Each row has 6 symbols:
% row1: 1..6 -> columns 1..6
% row2: 1..6 -> columns 7..12
P2 = P(:, 7:12);
P2n = mapminmax('apply', P2, ps);

Y2 = mlp_net(P2n);
[~, b2] = max(Y2);

atsakymas = [];
for k = 1:length(b2)
    atsakymas = [atsakymas, char('0' + b2(k))];  % 1->'1' ... 6->'6'
end

disp('MLP validation result:');
disp(atsakymas);
figure, text(0.1,0.5,atsakymas,'FontSize',38), axis off;

%% ================================================================
%  TESTING ON EXTERNAL IMAGES (MLP)
% ================================================================

test_files = {'test_data1.png','test_data2.png'};

for t = 1:numel(test_files)
    pavadinimas = test_files{t};

    pozymiai_patikrai = pozymiai_raidems_atpazinti(pavadinimas, 1);
    Ptest = cell2mat(pozymiai_patikrai);
    Ptestn = mapminmax('apply', Ptest, ps);

    Ytest = mlp_net(Ptestn);
    [~, btest] = max(Ytest);

    atsakymas = [];
    for k = 1:length(btest)
        atsakymas = [atsakymas, char('0' + btest(k))];
    end

    disp(['MLP Test ', num2str(t), ' (', test_files{t}, '):']);
    disp(atsakymas);

    figure, text(0.1,0.5,atsakymas,'FontSize',38), axis off;
end
