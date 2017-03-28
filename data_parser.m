% data parser 
% parses parkinson's data and splits into test and training data
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
% get feature names
data_file = fopen('data/parkinsons_data.csv', 'rt');
features = fgetl(data_file);
fclose(data_file);
sigma = 0.05;
% readcsv, split into training and test 
data = csvread('data/parkinsons_data.csv',1,0);
num_samples = 3;

% data read in (5875,22)
training_data = data(((num_samples/10)):end,:);
test_data = data(1:(num_samples/10),:);

% get y labels 
training_motor_updrs = training_data(:,5);
test_motor_updrs = test_data(:,5);
training_total_updrs = training_data(:,6);
test_total_updrs = test_data(:,6);

%get relevent variables 
x = training_data(:,7:end)';
x_star = test_data(:,7:end)';

% % parameters of K
lambda      = 1;   % output scale
ell         = 1;   % length scale

theta.cov = [log(ell); log(lambda)];
prior_mu = mean(x);
prior_K = covSEiso(theta.cov,x');
prior_K = (prior_K + prior_K')/2;
samples = mvnrnd(prior_mu, prior_K, num_samples);

plot(samples)
