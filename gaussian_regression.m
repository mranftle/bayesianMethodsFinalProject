format short g
% data parser 
% parses parkinson's data and splits into test and training data
% performs gaussian process regression with covSEiso kernel

sample_colors = ...
    [228,26,28; ...
    55,126,184; ...
    77,175,74; ...
    152,78,1631] / 255;

% get feature names for reference
data_file = fopen('data/parkinsons_data.csv', 'rt');
features = fgetl(data_file);
fclose(data_file);

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
x = training_data(:,7:end);
x_star = test_data(:,7:end);
n = size(x,1);
n_star = size(1000,1);

% noise 
sigma = 0.1;

% parameters of K
lambda      = 1;   % output scale
ell         = 1;   % length scale

theta.cov = [log(ell); log(lambda)];

% prior 
prior_mu = zeros(1,n_star);
prior_K = covSEiso(theta.cov,x_star);

%kernel 
% Kxx = covSEiso(theta.cov, x); 
Kxx = covSEiso(theta.cov,x);
% Kxs = covSEiso(theta.cov, x, x_star); 
Kxs = covSEiso(theta.cov,x,x_star);
% Kss = covSEiso(theta.cov, x_star); 
Kss = covSEiso(theta.cov,x_star);

% get posterior
V = Kxx + sigma^2*eye(n);
posterior_mu = (Kxs'/V) * training_total_updrs;
posterior_K = Kss - Kxs'/V * Kxs;

% root mean squared error
RMSE = sqrt(mean((posterior_mu - test_total_updrs).^2));
disp(RMSE);

% marginal liklihood
data_fit = ((training_total_updrs'/V)*training_total_updrs)/2;
disp(det(V));


% % plot posterior_mu and posterior_K
% figure; 
% x_sig = linspace(0,n_star,n_star)';
% sigma_h = ...
%     fill([x_sig; flipud(x_sig)],...
%         [posterior_mu - 2*sqrt(diag(posterior_K)); ...
%          flipud(posterior_mu + 2 * sqrt(diag(posterior_K)))], ...
%          sample_colors(3,:), ...
%          'edgecolor','none',...
%          'facealpha',0.3);
% hold('on');
% mean_h = ...
%     plot(posterior_mu, ...
%          'color', sample_colors(1, :));