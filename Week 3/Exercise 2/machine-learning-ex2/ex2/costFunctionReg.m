function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% Lorenzo Feliz

% hypothesis
h = sigmoid( X * theta );
% h dimension (m x 1)

% cost function
J = ((-1/m) * sum((y' * log(h)) + ((1 - y)' * log(1 - h)))) +  (lambda/(2*m)) * sum(theta(2:end).*theta(2:end));

% first order derivative of J

% for first term, (we calculate for the complete matrix the fist order
% differential)
grad = (1/m) * (X' * (h - y));

% now we will add the labda values starting from the 2nd index
grad(2:end) = grad(2:end) + (theta(2:end) .* (lambda/m)); 



% =============================================================

end
