function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));
thetaFiltered = [0; theta(2:end)];
g = sigmoid(X * theta);
J = (1/m)*sum((-y .* log(g)) - ((1-y) .* log(1-g))) + (lambda / (2 * m)) * sum(theta.^2) - (lambda / (2 * m)) * theta(1)^2;
grad = (1/m) *( X' * (sigmoid(X * theta) - y) ) + ( (lambda  * thetaFiltered) / m );

end
