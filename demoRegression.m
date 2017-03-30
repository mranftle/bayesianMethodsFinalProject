data_file = fopen('data/parkinsons_data.csv', 'rt');
features = fgetl(data_file);
fclose(data_file);

% readcsv, split into training and test (5875,22)
data = csvread('data/parkinsons_data.csv',1,0);
num_samples = length(data);
training_data = data(1:500,:);
test_data = data(501,:);

% get y labels 
training_motor_updrs = training_data(:,5);
test_motor_updrs = test_data(:,5);
y = training_data(:,6);
y_star = test_data(:,6);

%get relevent variables 
x = training_data(:,7:end);
x_star = test_data(:,7:end);


% x = gpml_randn(0.8, 20, 10);              % 20 training inputs
% y = sin(3*(sum(x,2))) + 0.1*gpml_randn(0.9, 20, 1);  % 20 noisy training targets
% xs = linspace(-20, 20, 500)';                  % 61 test inputs 

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y)

[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x_star);
RMSE = sqrt(mean((mu - y_star).^2))

mu
y_star
% f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
% fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
% hold on; plot(x_star, mu); 
% plot(x, y, '+')