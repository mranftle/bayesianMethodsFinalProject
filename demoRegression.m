data_file = fopen('data/parkinsons_data.csv', 'rt');
features = fgetl(data_file);
fclose(data_file);

% readcsv, split into training and test (5875,22)
data = csvread('data/parkinsons_data.csv',1,0);
num_samples = length(data);
training_data = data(1:500,:);
test_data = data(1001:end,:);

% get y labels 
training_motor_updrs = training_data(:,5);
test_motor_updrs = test_data(:,5);
y = training_data(:,6);
y_star = test_data(:,6);

%get relevent variables 
x = training_data(:,7:end);
x_star = test_data(:,7:end);

% training = csvread('numerai_datasets/numerai_training_data.csv',1,0);
% x = training(1:1000, 1:50);
% y = training(1:1000, 51);
% x_star = training(1001:5000, 1:50);
% y_star = training(1001:5000, 51);
 
D = size(x,2);
ell = 0; sf = 0;
meanfunc = [];                    % empty: don't use a mean function
cgi = {'covSEiso'};  hypgi = log([ell;sf]);    % isotropic Gaussian
cgu = {'covSEisoU'}; hypgu = log(ell);   % isotropic Gauss no scale
cla = {'covLINard'}; L = rand(D,1); hypla = log(L);  % linear (ARD)
cra = {'covRQard'}; al = 2; hypra = log([L;sf;al]); % ration. quad.
cca = {'covPPiso',2}; hypcc = hypgi;% compact support poly degree 2

likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, cgi, likfunc, x, y);

% nlml = gp(hyp, @infGaussLik, meanfunc, cgi, likfunc, x, y)

[mu s2] = gp(hyp2, @infGaussLik, meanfunc, cgi, likfunc, x, y, x_star);
RMSE = sqrt(mean((mu - y_star).^2));

mu;
y_star;
% f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
% fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
% hold on; plot(x_star, mu); 
% plot(x, y, '+')