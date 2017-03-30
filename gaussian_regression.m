format short g
% data parser 
% parses parkinson's data and splits into test and training data
% performs gaussian process regression with covSEiso kernel

sample_colors = ...
    [252, 146, 114; ...
     251, 106,  74; ...
     239,  59,  44; ...
     203,  24,  29; ...
     165,  15,  21] / 255;
 
% plot colors (from http://colorbrewer2.org/)
colors = [ 31, 120, 180; ...
           51, 160,  44; ...
          227,  26,  28; ...
          166, 206, 227] / 255;

num_samples = 1;
% get feature names for reference
data_file = fopen('data/parkinsons_data.csv', 'rt');
features = fgetl(data_file);
fclose(data_file);

disp(features);

% readcsv, split into training and test (5875,22)
data = csvread('data/parkinsons_data.csv',1,0);
training_data = data(((size(data,1)/10)):end,:);
test_data = data(1:(size(data,1)/10),:);

% get y labels 
training_motor_updrs = training_data(:,5);
test_motor_updrs = test_data(:,5);
training_total_updrs = training_data(:,6);
test_total_updrs = test_data(:,6);

% get relevent variables 
% x = training_data(:,7:end);
x = training_data(:,7:end);
x_star = test_data(:,7:end);
n_star = size(x,2);

%  prior
% prior_mu = zeros(1,n_star);
% prior_K = covZero(x);
% samples = mvnrnd(prior_mu, prior_K, num_samples)';

% % K(X, X)
Kxx = covSEiso(theta.cov,x);
% K(X, X_*)
Kxs = covSEiso(theta.cov,x,x_star);
% K(X_*, K_*)
Kss = covSEiso(theta.cov,x_star);

% posterior distribution for y*
posterior_mu = Kxs' / Kxx * training_total_updrs;
posterior_K  = Kss - Kxs' / Kxx * Kxs;
posterior_K  = (posterior_K + posterior_K') / 2;

RMSE = sqrt(mean((posterior_mu - test_total_updrs).^2)); 
disp(RMSE);