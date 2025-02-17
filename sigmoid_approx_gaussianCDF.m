% MATLAB Script to Approximate the Sigmoid Function using Gaussian CDF
% This script fits scalars a and b to minimize the Mean Squared Error (MSE)
% between the sigmoid function and a scaled and shifted Gaussian CDF over
% the range -T to T. It generates plots for different values of T.

% Clear all variables and close all figures
clear all; close all; clc;

% Define the range of T values to investigate
T_values = [0.5, 1, 5, 10, 100];

% Loop over each T value
for idx = 1:length(T_values)
    % Current T value
    T = T_values(idx);
    
    % Generate 1000 equally spaced points between -T and T
    z = linspace(-T, T, 1000);
    
    % Define the sigmoid function
    sigmoid = @(z) 1 ./ (1 + exp(-z));
    
    % Define the standard Gaussian CDF function (normcdf is built-in)
    gaussian_cdf = @(z) normcdf(z);
    
    % Objective function to minimize (Mean Squared Error between sigmoid and Gaussian CDF)
    % p(1) corresponds to 'a' and p(2) corresponds to 'b'
    error_function = @(p) mean((sigmoid(z) - gaussian_cdf(p(1)*z + p(2))).^2);
    
    % Initial guess for parameters [a, b]
    initial_guess = [1, 0];
    
    % Use fminsearch to find the optimal 'a' and 'b' that minimize the MSE
    options = optimset('Display', 'off');  % Suppress output
    [optimal_params, mse_value] = fminsearch(error_function, initial_guess, options);
    
    % Extract the optimized 'a' and 'b' values
    a = optimal_params(1);
    b = optimal_params(2);
    
    % Compute the Gaussian CDF approximation using the optimized parameters
    sigmoid_approx = gaussian_cdf(a * z + b);
    
    % Calculate the absolute error between the sigmoid and its approximation
    absolute_error = abs(sigmoid(z) - sigmoid_approx);
    
    % Plot the sigmoid function and its Gaussian CDF approximation
    figure;
    plot(z, sigmoid(z), 'b-', 'LineWidth', 2);          % Plot original sigmoid in blue
    hold on;
    plot(z, sigmoid_approx, 'r--', 'LineWidth', 2);     % Plot approximation in red dashed line
    xlabel('z');
    ylabel('Function Value');
    title(['Sigmoid vs. Gaussian CDF Approximation (T = ' num2str(T) ')']);
    legend('Sigmoid Function', 'Gaussian CDF Approximation');
    
    % Display the optimized parameters 'a' and 'b' on the plot
    text(0.1*T, 0.2, ['a = ' num2str(a)], 'FontSize', 12);
    text(0.1*T, 0.1, ['b = ' num2str(b)], 'FontSize', 12);
    
    % Plot the absolute error over the range -T to T
    figure;
    plot(z, absolute_error, 'k-', 'LineWidth', 2);      % Plot absolute error in black
    xlabel('z');
    ylabel('Absolute Error');
    title(['Absolute Error (T = ' num2str(T) ')']);
    
    % Calculate and display the Mean Absolute Error (MAE) in the legend
    mean_abs_error = mean(absolute_error);
    legend(['Mean Absolute Error = ' num2str(mean_abs_error)]);
    
end
