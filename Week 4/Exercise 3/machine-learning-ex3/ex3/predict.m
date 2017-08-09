function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Lorenzo Feliz

% layer 1 : 400 units + 1 bias unit (only used for calculations for next layer)(input layer)
% layer 2 : 25 units + 1 bias unit (only used for calculations for next layer)
% layer 3 : 10 units (output layer)

% Theta1 : dimension 25 x 401
% Theta2 : dimension 10 x 26
% X : dimension m x 400

%% Layer 1 : Input layer
% adding bais unit, new column where all the elements are 1 added
a1 = [ones(m, 1) X]; % dimensions m x 401

%% Layer 2 : Hidded layer
% Activation Calculations for second layer 
z2 = a1 * Theta1'; % dimensions m x 25

% Activation values 
a2 = sigmoid(z2); % dimension m x 25

%% Layer 3 : Output Layer
% Adding the bias unit on second layer units
a2 = [ones(m, 1) a2]; % dimensions m x 26

% Activation Calculations for third layer
z3 = a2 * Theta2'; % dimension m x 10 

% Activation Values
a3 = sigmoid(z3); % dimension m x 10

% Selecting the index value with highest prediction as the predicted output
% from the Neural Network
[value, index] = max(a3, [],2);
p = index;

% =========================================================================


end
