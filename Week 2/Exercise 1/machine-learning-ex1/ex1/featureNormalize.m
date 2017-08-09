function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% mean for each of the column in matrix X is given below, also mu will be a
% row vector , which means just a single row and n number of columns where
% n is the number of features.

% size of mean(X) is 1 X number of features in X,(1 X n) %% Lorenzo Feliz
mu = mean(X);

% for each of the feature we need the difference between that corresponding
% feature's minimum element and the maximum element, we shall be using the
% following mehod for that, Also note that min and max functions returns
% row metrics 

% size of diffMaxMin is 1 X number of features in X ,(1 X n) %% Lorenzo Feliz
% diffMaxMin = max(X) - min(X);
% sigma = diffMaxMin; 
% This above method does not give proper answers when normalising 

% Method suggested in the ex1.pdf, using std deviation inplace of max - min for calculating Sigma 
sigma = std(X);

% Temporary column vector with number of rows equal to number of rows in X,
% which when multiplied with row vectors like 
% mean(X) we would get m X n matrix, where m is the number of rows present
% in X and n is number of features present in X. Also after multiplication
% each column of the resulting matric shall have same value.

% size of t is number of rows in 'X' X 1 ,(m X 1) %% Lorenzo Feliz
t = ones(length(X),1);

% Now we calculate normalised feaure matrics for X as ,%% Lorenzo Feliz

X_norm = (X - (t*mu)) ./ (t*sigma);

% ============================================================

end
