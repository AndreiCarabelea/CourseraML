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

for j = 1 : m
   h = sigmoid(X(j,:)*theta)
   yj = y(j,:)
   Jj = (-yj)*log(h) - (1-yj)*log(1-h) 
   Jj = Jj/m
   J = J + Jj;

%extract first row from theta and store in theta1
theta1 = theta
theta1(1) = 0;

J = J + (sum(theta1.^2))*(lambda/(2*m)); 


errors = sigmoid(X*theta)-y;
grad = X'*errors ;
grad = grad/m;

regTerm = (lambda/m).*theta
regTerm(1)=0

grad = grad + regTerm


% =============================================================

end
