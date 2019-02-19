% normal boundary

clear;
addpath('/ssd1/xchen/sporco/Util');

N = 30;

%load('./sgd_D12_M64_lam0.05.mat');
load('/home/xhchrn/DeepBregmanISS/conv_dicts/dicts_voc2012/sgd_D7_M64_lam0.05.mat');
% load('./sgd_D16_M100_lam0.05.mat');

Ds = size(D,1);
M = size(D,3);
% tic;
Dop = Doperator_sparse(D,N);
% toc
% tic;
% D2 = Op2D(Dop,Ds,M,N);
% toc
% norm(D(:)-D2(:))
% norm(D(:))
% norm(D2(:))

maxit = 200;
W = Dop;
eta = 0.002;
for t = 1:maxit
    res = Dop' * W;
    
    f = sum(sum(res.*res));
%     f = norm(res(res~=0))^2;
    fprintf('t: %d\t f: %.3f\n',t,full(f));
    
    gra = Dop * res;
    W = W - eta * gra;
    W = Op2D(W,Ds,M,N);
    W = proj(W,D);
    W = Doperator_sparse(W,N);
end
W = Op2D(W,Ds,M,N);
imdisp(tiledict(W));

%% functions

function Dop = Doperator_sparse(D,N)

Ds = size(D,1);
M = size(D,3);
myzpad = @(D) padarray(D,[N-1,N-1],'post');

triple = zeros(N^2*Ds^2*M,3);
num=0;
for m = 1:M
%     m_index = (1+(m-1)*N^2):(N^2+(m-1)*N^2);
    for n = 1:N^2
        j = ceil(n/N);
        i = mod(n,N);
        if i == 0, i=N; end
        
        Dpnow = sparse(double(myzpad(D(:,:,m))));
        Dpnow = reshape(circshift(Dpnow, [i-1,j-1]),[(N+Ds-1)^2,1]);
        [ii,~,ss] = find(Dpnow);
        num_now = size(ii,1);
        ii = ii + (m-1)*(N+Ds-1)^2;
        triple(num+1:num+num_now,:)=[repmat(n,[num_now,1]),ii,ss];
        num = num + num_now;
    end
end
triple = triple(1:num,:);
Dop = sparse(triple(:,1),triple(:,2),triple(:,3),N^2,(N+Ds-1)^2*M);
end

function Dop = D2op_sp(D,N,Ds)
triple = zeros(N^2,3);

for n = 1:N^2
	j = ceil(n/N);
	i = mod(n,N);
	if i == 0, i=N; end
        
	Dpnow = reshape(circshift(D, [i-1,j-1]),[(N+Ds-1)^2,1]);
	[ii,~,ss] = find(Dpnow);
	triple(n,:)=[n,ii,ss];
end
Dop = sparse(triple(:,1),triple(:,2),triple(:,3),N^2,(N+Ds-1)^2);
end

function D = Op2D(Dop,Ds,M,N)

Dops = cell(1,M);
for m = 1:M
    m_index = (1+(m-1)*(N+Ds-1)^2):((N+Ds-1)^2+(m-1)*(N+Ds-1)^2);
    Dops{m} = Dop(:,m_index);
end
D = zeros(Ds,Ds,M);
for m = 1:M
    for i = 1:Ds
        for j = 1:Ds
            Dnow = sparse(i,j,1,N+Ds-1,N+Ds-1);
            Dnow_op = D2op_sp(Dnow,N,Ds);
            Dnow_op = Dnow_op .* Dops{m};
            ss = sum(Dnow_op(:))/N/N;
            D(i,j,m) = ss;
        end
    end
end

end


function Wout = proj(Win,D)
M = size(D,3);
N = size(D,1);
A = zeros(N*N,M);
W = zeros(N*N,M);
for ii = 1:M
    A(:,ii) = reshape(D(:,:,ii),[N*N,1]);
    W(:,ii) = reshape(Win(:,:,ii),[N*N,1]);
end
aw = diag(A'*W);
aw = repmat(aw',[size(A,1), 1]);
W_next = W + (1-aw).*A;
Wout = zeros(N,N,M);
for ii = 1:M
    Wout(:,:,ii) = reshape(W_next(:,ii),[N,N]);
end
end


