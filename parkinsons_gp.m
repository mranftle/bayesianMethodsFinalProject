data_file = fopen('data/parkinsons_data.csv', 'rt');

% get features 
features = strsplit(fgetl(data_file), ','); 
fclose(data_file); 

% readcsv, get training and test (5875, 22) 
data = csvread('data/parkinsons_data.csv',1,0);
num_samples = length(data);
training_data = data(find(data(:,1) < 37),:);
test_data = data(find(data(:,1) >= 37), :);

% sample random points from training data 
rand_train_sample = randsample(1:length(training_data), 2000);
training_data = sortrows(training_data(rand_train_sample,:), 5);

rand_test_sample = randsample(1:length(test_data),50);
test_data = sortrows(test_data(rand_test_sample, :), 5);

x = training_data(:,7:end); 
y = training_data(:,6);  % motor updrs 
x_star = test_data(:,7:end);
y_star = test_data(:,6); % motor upd

% pca 
% [eigenvectors, projected_data, eigenvalues] = princomp(x);
% [foo, feature_idx] = sort(eigenvalues, 'descend');
% x = x(:, feature_idx(1:10));
% x_star = x_star(:, feature_idx(1:10));

% parameters 
input_scale = 1; 
output_scale = 1; 
noise = 0.5; 

% define GP model 

% prior mean = 0
mean_function = {@meanZero}; 

% prior covariance
% K(x, x'; ?, ?) = ?² exp(-|x - x'|² / 2?²)
covar_function = {@covRQiso} ;

% hyperparameters 
hyperparameters.mean = []; 
hyperparameters.cov = ...
    [log(input_scale); ...
    log(output_scale)]; 
hyperparameters.lik = log(noise); 

% prior p(y | X, theta) 
mu = feval(mean_function{:}, hyperparameters.mean, x); 

% covariance of f(X), K(X, X) 
K_f = feval(covariance_function{:}, hyperparameters.cov, x);

% covariance of y(X), K(X, X) + ?² I
K_y = K_f + noise^2 * eye(length(x)); 

% disp(K_f);
% samples = mvnrnd(mu, K_y)'; 
% figure(1);
% hold('off');
% plot(x, samples, '.');

training_nml = gp(hyperparameters, [], mean_function, covariance_function, ...
                  [], x, y);

learned_hyperparameters = minimize(hyperparameters, @gp, 100, [], ...
        mean_function, covariance_function, [], x, y);

[y_mean, y_var] = gp(learned_hyperparameters, [], mean_function, ...
    covariance_function, [], x, y, x_star);

hold('on');
disp(cov(x));
plot(y_mean, 'r');
plot(y_star, 'b');
plot(y_mean + 2 * sqrt(y_var), 'g');
plot(y_mean - 2 * sqrt(y_var), 'g');
