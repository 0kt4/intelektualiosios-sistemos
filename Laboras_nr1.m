% Image file names for apples and pears (Training set)
appleImages = {'apple_04.jpg', 'apple_05.jpg', 'apple_06.jpg'};
pearImages = {'pear_01.jpg', 'pear_02.jpg'};

% Initialize feature vectors for color and roundness
x1 = zeros(1, 5); % Color feature
x2 = zeros(1, 5); % Roundness feature

% Process apple images (3 apples)
for i = 1:3
    img = imread(appleImages{i});
    x1(i) = spalva_color(img);    % Extract color feature
    x2(i) = apvalumas_roundness(img);  % Extract roundness feature
end

% Process pear images (2 pears)
for i = 1:2
    img = imread(pearImages{i});
    x1(3 + i) = spalva_color(img);    % Extract color feature
    x2(3 + i) = apvalumas_roundness(img);  % Extract roundness feature
end

% Estimated features are stored in matrix P
P = [x1; x2];

% Desired output vector (1 for apples, -1 for pears)
T = [1; 1; 1; -1; -1];

%% Train single perceptron with two inputs and one output

% Generate random initial values of w1, w2, and b
w1 = randn(1);
w2 = randn(1);
b = randn(1);

% Learning rate
eta = 0.1;

% Initialize total error to a large value
e = 1;

% Training loop - continue until the total error is 0
while e ~= 0
    % Initialize total error to 0 for each iteration
    total_error = 0;
    
    % Loop through each input example
    for i = 1:5
        % Calculate the weighted sum (v) for current example
        v = w1 * x1(i) + w2 * x2(i) + b;
        
        % Calculate the perceptron output (y)
        if v > 0
            y = 1;
        else
            y = -1;
        end
        
        % Calculate the error (desired output - actual output)
        e = T(i) - y;
        
        % Update the weights and bias if there is an error
        w1 = w1 + eta * e * x1(i); % Update w1
        w2 = w2 + eta * e * x2(i); % Update w2
        b = b + eta * e;           % Update bias
        
        % Accumulate total error
        total_error = total_error + abs(e);
    end
    
    % Update the total error for stopping condition
    e = total_error;
end

% Output the final weights and bias after training
fprintf('Final weights and bias after perceptron training:\n');
fprintf('w1: %.4f\n', w1);
fprintf('w2: %.4f\n', w2);
fprintf('b: %.4f\n', b);

%% Test the perceptron with new images
newImages = {'apple_07.jpg','apple_11.jpg','pear_03.jpg', 'pear_09.jpg'}; % Add more images as needed

for i = 1:length(newImages)
    % Load and process new image
    new_img = imread(newImages{i});
    
    % Extract features for the new image (color and roundness)
    new_x1 = spalva_color(new_img);    % Extract color feature
    new_x2 = apvalumas_roundness(new_img);  % Extract roundness feature
    
    % Calculate the weighted sum (v) for the new image
    v_new = w1 * new_x1 + w2 * new_x2 + b;
    
    % Calculate the perceptron output (y) for the new image
    if v_new > 0
        y_new = 1; % Classified as apple
    else
        y_new = -1; % Classified as pear
    end
    
    % Display the classification result
    if y_new == 1
        fprintf('Perceptron: The image "%s" is classified as an apple.\n', newImages{i});
    else
        fprintf('Perceptron: The image "%s" is classified as a pear.\n', newImages{i});
    end
end

%% Naive Bayes Classifier

% Step 1: Calculate Priors
numApples = sum(T == 1);
numPears = sum(T == -1);
numTotal = length(T);

% Prior probabilities
P_apple = numApples / numTotal;
P_pear = numPears / numTotal;

% Step 2: Calculate Likelihoods
mean_color_apple = mean(x1(1:3)); % Mean of color for apples
std_color_apple = std(x1(1:3));   % Std dev of color for apples

mean_color_pear = mean(x1(4:5));  % Mean of color for pears
std_color_pear = std(x1(4:5));    % Std dev of color for pears

mean_roundness_apple = mean(x2(1:3)); % Mean of roundness for apples
std_roundness_apple = std(x2(1:3));   % Std dev of roundness for apples

mean_roundness_pear = mean(x2(4:5));  % Mean of roundness for pears
std_roundness_pear = std(x2(4:5));    % Std dev of roundness for pears

% Gaussian likelihood function
gaussian_likelihood = @(x, mean, std) (1 / (std * sqrt(2 * pi))) * exp(-(x - mean)^2 / (2 * std^2));

% Step 3: Test with the same new images using Naive Bayes
for i = 1:length(newImages)
    % Load and process new image
    new_img = imread(newImages{i});
    
    % Extract features for the new image (color and roundness)
    new_x1 = spalva_color(new_img);    % Extract color feature
    new_x2 = apvalumas_roundness(new_img);  % Extract roundness feature
    
    % Calculate likelihoods for the new image
    % Likelihood for apple class
    L_color_apple = gaussian_likelihood(new_x1, mean_color_apple, std_color_apple);
    L_roundness_apple = gaussian_likelihood(new_x2, mean_roundness_apple, std_roundness_apple);
    P_apple_given_features = P_apple * L_color_apple * L_roundness_apple;
    
    % Likelihood for pear class
    L_color_pear = gaussian_likelihood(new_x1, mean_color_pear, std_color_pear);
    L_roundness_pear = gaussian_likelihood(new_x2, mean_roundness_pear, std_roundness_pear);
    P_pear_given_features = P_pear * L_color_pear * L_roundness_pear;
    
    % Classify the image based on the higher posterior probability
    if P_apple_given_features > P_pear_given_features
        fprintf('Naive Bayes: The image "%s" is classified as an apple.\n', newImages{i});
    else
        fprintf('Naive Bayes: The image "%s" is classified as a pear.\n', newImages{i});
    end
end
