clear;

%% initialization
%rng(0);
m = 250;
n = 500;
A = randn(m,n); % good A
%A = ones(m,n) + randn(m,n) * 0.1; % bad A
for i = 1:n
    A(:,i) = A(:,i)/norm(A(:,i));
end

%% iteration
W = A;
f = func(A,A);
maxit = 5000;
gamma0 = 1;
c = 0.5;
tau = 0.5;
Q = ones(n,n)+eye(n)*(n-2);
Ainv = (A * A')^(-1);
for t = 1: maxit
    res = A' * W - eye(n);
    gra = A * ( Q .* nabla_f(res));
    dir = Ainv * gra;
    innerprod = sum(sum(dir.*gra));
    gamma = gamma0;
    for attempt = 1:1e5
        W_next = W - gamma * dir;
        f_next = func(W_next,A);
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
function f = func(W,A)
n = size(A,2);
res = A' * W - eye(n);
Q = ones(n,n)+eye(n)*(n-2);
res = res .* sqrt(Q);
f = sum(sum(exp(res.*res))); 
end

function g = nabla_f(B)
g = 2 * B .* exp(B.*B); 
end


