function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));
g = sigmoid(X * theta);
J = (1/m)*sum((-y .* log(g)) - ((1-y) .* log(1-g)));
grad = (1/m) *( X' * (sigmoid(X * theta) - y) );

% Note: grad should have the same dimensions as theta

end
