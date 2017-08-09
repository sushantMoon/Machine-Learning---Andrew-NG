function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Lorenzo Feliz

% y vector needs to be changed 
yMatrix = zeros(m,num_labels);  % dimensions m x k (k beign the number of classes)
for i = 1:m
    yMatrix(i,y(i)) = 1;
end

% Part 1 
% Coded as guided in 1.3 in the ex4.pdf
% Implementation for Cost function without regularization

% We need activation values for ouput layer units { hTheta(x),
% i.e the final predicted value from each of the unit in final output layer 
% are equal to final layer's activation values ) }

% copied the the following lines from predict.m as it also performs the
% similar operations
h1 = sigmoid([ones(m, 1) X] * Theta1');  % a2
h2 = sigmoid([ones(m, 1) h1] * Theta2'); % a3

% h2 is the vector that has activation values for output layer units
hThetaX = h2;   % the log of this value is going to be used in the cost function 
% dimensions m x k (k beign the number of classes) 

% cost funtion without regularization
J = (1/m) * sum(sum((-1 * yMatrix .* log(hThetaX)) - ((ones(m,num_labels) - yMatrix) .* log(ones(m,num_labels) - hThetaX))));

% Adding Regularization on this cost function 
% Coded as guided in 1.4 in the ex4.pdf
regularizationValues = (lambda / (2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = J + regularizationValues;

% Part 2
% Coded as guided in 2.3 in the ex4.pdf
% We will be iterating over the training setone by one and following the
% back prop algorithm as mentioned

for t = 1:m
    
    % Forwards propogation
    % Input layer 
    a1 = [1; X(t,:)'];      % dimension 401 x 1
    % First Hidden Layer
    z2 = Theta1 * a1;      % dimension 25 x 1
    a2 = [1; sigmoid(z2)];      % dimension 26 x 1
    % Output Layer
    z3 = Theta2 * a2;      % dimension 10 x 1
    a3 = sigmoid(z3);       % dimension 10 x 1
    
    % Small Delta Calculation
    % Error for Units in layer 3
    delta3 = a3 - (yMatrix(t,:)');      % dimension 10 x 1
    % Error for Units in layer 2
    delta2 = ((Theta2') * delta3) .* [1; sigmoidGradient(z2)];      % dimension 26 x 1
    % Removing the delta for bias row
    delta2 = delta2(2:end);      % dimension 25 x 1
    
    % Large Delta Calculations (Accumulators)
    Theta1_grad = Theta1_grad + (delta2 * a1');     % dimension 25 x 401
    Theta2_grad = Theta2_grad + (delta3 * a2');     % dimension 10 X 26
end

% Unregularised differential of cost function
Theta1_grad = (1/m) * Theta1_grad;      % dimension 25 x 401
Theta2_grad = (1/m) * Theta2_grad;     % dimension 10 X 26

% Regularised values for differential of cost function 
% First part zeros(size(Theta1,1),1) corresponds to the j = 0 case where we
% do not regularise the parameter for bias row
reg_Theta1_grad = (lambda/m) * [zeros(size(Theta1,1),1) Theta1(:,2:end)];
reg_Theta2_grad = (lambda/m) * [zeros(size(Theta2,1),1) Theta2(:,2:end)];

Theta1_grad =  Theta1_grad + reg_Theta1_grad;
Theta2_grad =  Theta2_grad + reg_Theta2_grad;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
