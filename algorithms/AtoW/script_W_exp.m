clear;

%% initialization: generating A
% rng(0); % random seed
m = 250;
n = 500;
A = randn(m,n); % good A
%A = ones(m,n) + randn(m,n) * 0.1; % bad A
for i = 1:n
    A(:,i) = A(:,i)/norm(A(:,i));
end

%% iteration
alpha = 1;
% note: if A is bad, alpha is large, the algorithm may be slow.

W = A;
f = func(A,A,alpha);
maxit = 5000;
gamma0 = 1;
c = 0.5;
tau = 0.5;
Ainv = (A * A')^(-1); % precondition
for t = 1: maxit
    res = A' * W - eye(n);
    gra = A * nabla_f(res,alpha);
    dir = Ainv * gra;
    innerprod = sum(sum(dir.*gra));
    gamma = gamma0;
    for attempt = 1:1e5 % line search
        W_next = W - gamma * dir;
        f_next = func(W_next,A,alpha);
        % Armijo backtracking
        if f_next - f < - gamma * c * innerprod
            W = W_next;
            f = f_next;
            break;
        end
        gamma = tau * gamma;
    end
    if norm(gra(:))/norm(W(:))<1e-3, break; end
    fprintf('t: %d\t, attempts:%d\t, func: %f\n', t, attempt, f);
end

%% display
visualizationW(A,W);

%% functions
function f = func(W,A,sigma)
res = A' * W - eye(size(A,2));
f = sum(sum(exp(sigma*res.*res)));
end

function g = nabla_f(B, sigma)
g = 2 * B .* exp(sigma*B.*B); 
end


