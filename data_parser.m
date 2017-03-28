% data parser 
% parses parkinson's data and splits into test and training data

% get feature names
data_file = fopen('data/parkinsons_data.csv', 'rt');
features = fgetl(data_file);
fclose(data_file);

% readcsv, split into training and test 
data = csvread('data/parkinsons_data.csv',1,0);
num_samples = length(data);
test_data = data(1:(num_samples/10),:);
training_data = data(((num_samples/10)):end,:);

% Gaussian Process Regression
motor_updrs =training_data(:, 5); % training motor updrs score
total_updrs = training_data(:, 6); % training total updrs score
avg_total_updrs = mean(total_updrs); % avg total updrs score, use are prior mu

disp(training_data);