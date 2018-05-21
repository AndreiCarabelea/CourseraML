function [J grad] = nnCostFunction(nn_params, ...
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

function result = expandY (label, k)
  result = zeros(k,1)
  result(label,:) = 1
endfunction

function total = sumOfSquares ( matrixV ) 
  % strip the first column 
  matrixVTemp = matrixV(:, 2:end)
  % square every element
  matrixVTemp = matrixVTemp .^2
  %sum all elements
  total = sum(sum(matrixVTemp))
endfunction
  
  


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
                 
             

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;

% obtain the matrix outpout 
% X is 5000 x 400 
% Theta1 is 25 x 401 
% Theta2 is 10  x 26
% y is 5000 x 1 
% z is computed by multiplying matrixes having 1 for bias units ( first column is 1 ) 
% 
% X is now 5000  x 401 , a bias unit , the examples are displayed on rows. 
X = [ ones(m,1) X];

% activation on first layer 25 x 5000
z2 = Theta1 * X';
y2 = sigmoid(z2);

% add a bias unit 26  x 5000
y2 = [ ones(1, m); y2];

% output layer 10 x 5000 , each training output on a column 
z3 = Theta2 * y2;

% we don't add a bias unit because this is the right most layer.
y3 = sigmoid(z3);


%binary output 
matY = zeros(num_labels, m);

for index = 1:m
  matY(:,index) = expandY(y(index, :), num_labels);
end  

% cost functions for each training example, column vector. 
jj = zeros(m, 1);

for i =1:m 
  T1 = dot(-matY(:,i), log(y3(:,i)));
  T2 = dot(1-matY(:,i), log(1 - y3(:,i)));
  jj(i,:) = T1 - T2; 
end

J = mean(jj) ;

J = J + (lambda/(2*m))*(sumOfSquares(Theta1) + sumOfSquares(Theta2));


Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Delta1_grad = zeros(size(Theta1));
Delta2_grad = zeros(size(Theta2));


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


% -------------------------------------------------------------

% =========================================================================


for index = 1:m
  
  %10 x 1 
  delta3 = y3(:,index) - matY(:,index)
 
  %25 x 1 
  gdz2 = z2(:, index);
  
  %25 x 1 
  gdz2 = sigmoidGradient(gdz2);
  
  %26 x 1 
  gdz2 = [1 ; gdz2];
  
  % 26 x 1 
  delta2 = (Theta2'*delta3).* gdz2;
  
  % 25 x 1 
  delta2 = delta2(2:end,:);
  
  % 10 x 26 
  Delta2_grad = Delta2_grad + delta3 * y2(:,index)';
  
  % 25 x 401
  Delta1_grad =  Delta1_grad + delta2 * X(index, :);
  
  
end 
   
  Theta1_grad = Delta1_grad/m
  Theta2_grad = Delta2_grad/m
  
  %theta1Reg = Theta1_grad(:, 2:end);
  %theta1Reg = theta1Reg * ( 1 + lambda );
  %Theta1_grad = [Theta1_grad(:,1) theta1Reg];
  
  %theta2Reg = Theta1_grad(:, 2:end);
  %theta2Reg = theta2Reg *( 1 + lambda);
  %Theta2_grad = [Theta2_grad(: , 1) theta2Reg];
   
  
  
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
