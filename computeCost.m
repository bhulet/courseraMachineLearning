% Compute the cost function for linear regression
function J = computeCost(X, y, theta)
m = length(y); % number of training examples
predictions = X*theta;
sqrErrors = (predictions - y).^2
J = 0;
J = 1/(2*m) * sum(sqrErrors);

end
