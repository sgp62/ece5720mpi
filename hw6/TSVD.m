%
% Assignment 6, matlab's implementation
%

% set A, 
A = [0 1 2; 3 4 5; 6 7 8; 9 10 11];
m = 4; n = 3;

% get the SVD of A
[U,S,V] = svd(A);

disp(diag(S)')
% 2.2447e+01   1.4641e+00   7.1624e-16
% so rank of A is k = 2
k = 2;
% construct Uk, Vk, Sk
%
Uk = U(:,1:k); Vk = V(:,1:k); Sk = S(1:k,1:k);
%
% approximate A by Ak = Uk*Sk*Vk'
Ak = Uk*Sk*Vk';

% see how well AK approximates A
Aerror = norm(A-Ak,'fro')

% set solution x to all ones
xt = ones(n,1);

% get the corresponding rhs vector
b = A*xt;

% see the error between "true" xt and "computed" xc
xc = A\b;
xerror = norm(xt-xc)/norm(xt)

% compute the residual error A*xc-b
reserror = norm(A*xc-b)

% get inverse of Sk 
invSk = 1./diag(Sk);

% now compute xtsvd form the TSVD
xtsvd = Vk*(invSk.*(Uk'*b));

% get the error 
xtsvderror = norm(xtsvd-xt)/norm(xtsvd)

% get the residual error
restsvderror = norm(A*xtsvd-b)
