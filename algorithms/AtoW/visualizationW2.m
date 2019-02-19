function visualizationW(A,W)

n = size(A,2);

res = A' * W - eye(n);
res0 = A' * A - eye(n);

figure ('Units', 'pixels', 'Position', [300 300 800 275]) ;

subplot(1,2,1);
histogram(res(~eye(n)),'BinWidth',2e-2);
hold on;
histogram(res0(~eye(n)),'BinWidth',2e-2);
title('off-diagonal');
legend('W^TA-I','A^TA-I');
hold off;

subplot(1,2,2);
histogram(res(logical(eye(n))),'BinWidth',1e-5);
hold on;
histogram(res0(logical(eye(n))),'BinWidth',1e-5);
hold off;
title('diagonal');

end
