function [ ] = mean_project( )
%PROJECT Summary of this function goes here
%   Detailed explanation goes here
% data parser 
% parses parkinson's data and splits into test and training data
% performs gaussian process regression with covSEiso kernel
    
% get feature names for reference
% data_file = fopen('data/parkinsons_data.csv', 'rt');
% features = fgetl(data_file);
% fclose(data_file);
% 
% % readcsv, split into training and test (5875,22)
% data = csvread('data/parkinsons_data.csv',1,0);
% num_samples = length(data);
% training_data = data(((num_samples/10)):end,:);
% size(training_data);
% test_data = data(1:(num_samples/10),:);
% 
% % get y labels 
% training_motor_updrs = training_data(:,5);
% test_motor_updrs = test_data(:,5);
% training_total_updrs = training_data(:,6);
% test_total_updrs = test_data(:,6);
% 
% %get relevent variables 
% x = training_data(:,7:end);
% x_star = test_data(:,7:end);

training = csvread('numerai_datasets/numerai_training_data.csv',1,0);
x = training(1:100, 1:50);
y = training(1:100, 51);
x_star = training(101:200, 1:50);
y_star = training(101:200, 51);
n = length(x);
n_star = length(x_star);

% parameters of K
lambda      = 1;   % output scale
ell         = 1;   % length scale
theta.cov = [log(ell); log(lambda)];
k=50;
s = 0.1;
phi = [ones(length(x),1)];
phi_star = [ones(length(x_star),1)];

mu = zeros(k+1,1);
sigma = eye(k+1);
phi = [phi x];
phi_star = [phi_star x_star];
mu_post1 = mu + sigma*phi'/(phi*sigma*phi'+s^2*eye(n))*(y_star-phi*mu);
sigma_post1 = sigma - sigma*phi'/(phi*sigma*phi'+s^2*eye(n))*phi*sigma;
mu_post_y = phi_star*mu_post1;
sigma_post_y = phi_star * sigma_post1 * phi_star' + s^2 * eye(n_star);
% fprintf('Marginal log-likelihood for k = %i = %0.4f\n', ...
%       k, ...
%       log_mvnpdf(training_motor_updrs, ...
%                  phi * mu, ...
%                  phi * sigma * phi' + s^2 * eye(5288)));
RMSE = sqrt(mean((mu_post_y - y_star).^2))
size(mu_post_y)
size(x_star)
mvnpdf(y_star,mu_post_y, sigma_post_y)
% figure
% mean_ = plot(mu_post_y);
% hold on
% % observations = plot(x, y, 'x');
% sigma = ...
%     fill([x_star; flipud(x_star)], ...
%          [mu_post_y - ...
%           2 * sqrt(diag(sigma_post_y));
%           flipud(mu_post_y + ...
%                  2 * sqrt(diag(sigma_post_y)))], ...
%          'blue', ...
%          'edgecolor', 'none', ...
%          'facealpha', 0.3);

end

