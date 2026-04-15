function [p] = gaussmmC(u, sigma2, y, d)
%caluculate gaussian emission probabilities
%input mean vector, here is 2D, sigmas
n = size(u,2); % state number
% y: rowvec, 1*2
% u: matrix: n*2
% sigma2: high-order matrix: n*2*2
% a: rowvec: n*1. Gauss mixture weights
p = zeros(n,1);
for i = 1:n % forloop of the states
    covm     = sigma2(:,:,i); 
    detcovm  = det(covm);
    
    covm_inv = inv(covm);
    exv = exp(-0.5*(y-u(:,i))'*covm_inv*(y-u(:,i)));
    p(i) = 1/((2*pi)^(d/2)*detcovm^0.5)*exv; 
end
end
