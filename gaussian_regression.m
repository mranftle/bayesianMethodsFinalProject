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

% % get y labels 
training_motor_updrs = training_data(:,5);
test_motor_updrs = test_data(:,5);
training_total_updrs = training_data(:,6);
test_total_updrs = test_data(:,6);

% %get relevent variables 
% x = training_data(:,7:end);
x = training_data(:,7:end)';
x_star = test_data(:,7:end)';

%  prior
prior_mu = zeros(1,size(x,1));
prior_K = covSEiso(theta.cov,x);
% prior_K = (prior_K + prior_K')/2;
% 
% samples = mvnrnd(prior_mu, prior_K, num_samples)';
% 
% % for i = 1:num_samples
% %   samples_h = ...
% %       plot(x_star, samples(:, i), ...
% %            'color', sample_colors(i, :));
% %   hold('on');
% % end
% % sigma_h = ...
% %     fill([x_star; flipud(x_star)], ...
% %          [prior_mu - 2 * sqrt(diag(prior_K));
% %           flipud(prior_mu + 2 * sqrt(diag(prior_K)))], ...
% %          colors(4, :), ...
% %          'edgecolor', 'none', ...
% %          'facealpha', 0.3);
% % mean_h = ...
% %     plot(x_star, prior_mu, ...
% %          'color', colors(1, :));
% % 
% % ylim([-5, 5]);
% % 
% % set(gca, 'box', 'off');
% % 
% % xlabel('$x$');
% 
% % legend([mean_h, sigma_h, samples_h], ...
% %        '$\mu(x)$',              ...
% %        '$\pm 2\sqrt{K(x, x)}$', r...
% %        'samples',               ...
% %        'location', 'northeast');
% % legend('boxoff');
% 
% % %
% % %PLOT OF PRIOR SAMPLES HERE 
% % 
% % % posterior
% % % using squared exponential kernel, loop here over multiple kernels
% % 
% % % K(X, X)
% 
% Kxx = covSEiso(theta.cov, x);
% % % K(X, X_*)
% Kxs = covSEiso(theta.cov, x, x_star);
% % % K(X_*, K_*)
% 
% Kss = covSEiso(theta.cov, x_star);
% % 
% % disp(size(Kxx));
% % disp(size(Kxs));
% % disp(size(Kss));
%  
% % posterior distribution for y*
% posterior_mu = Kxs' / Kxx * training_total_updrs;
% posterior_K  = Kss - Kxs' / Kxx * Kxs;
% posterior_K  = (posterior_K + posterior_K') / 2;
%  
% disp(size(posterior_mu));
% disp(size(posterior_K));
% disp(size(x_star));
% 
% disp(posterior_mu);
% disp(test_total_updrs);
% 
% RMSE = sqrt(mean((posterior_mu - test_total_updrs).^2)); 
% disp(RMSE);
% 
% figure;
% hold('off');
% sigma_h = ...
%     fill([x_star; flipud(x_star)], ...
%          [posterior_mu - 2 * sqrt(diag(posterior_K));
%           flipud(posterior_mu + 2 * sqrt(diag(posterior_K)))], ...
%          colors(4, :), ...
%          'edgecolor', 'none', ...
%          'facealpha', 0.3);
% hold('on');
% mean_h = ...
%     plot(x_star, posterior_mu, ...
%          'color', colors(1, :));
% observations_h = plot(x, training_total_updrs, 'k.');
% 
% ylim([-3, 3]);
% 
% set(gca, 'box', 'off');
% 
% xlabel('$x$');

% legend([mean_h, sigma_h, observations_h], ...
%        '$\mu(x)$',              ...
%        '$\pm 2\sigma$', ...
%        'observations',          ...
%        'location', 'northeast');
% legend('boxoff');

% 
% 
% %PLOT POSTERIOR SAMPLES HERE% 
% % 
% % %%%%how to improve? 
% % 
