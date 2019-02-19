clear;

%% initialization
%rng(0);
m = 250;
n = 500;
load ('A.mat');
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
    gra = A * ( Q .* res);
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

save ('W_rel2.mat', 'W');

%% display
visualizationW2(A,W);

%% functions
function f = func(W,A)
n = size(A,2);
res = A' * W - eye(n);
Q = ones(n,n)+eye(n)*(n-2);
res = res .* sqrt(Q);
f = sum(sum(res.*res));
end


