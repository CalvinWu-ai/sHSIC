p=150;
beta=[1,1,1,zeros(1,p-3)]'/sqrt(3);
psig=0.5.^abs((1:p)'-(1:p));
spsig = sqrtm(psig);
n=100; 

        
rng(10,'twister');

x=randn(n,p)*spsig;
y=1+exp(x*beta)+randn(n,1);
Pi=sHSIC(x,y,0.02);
[U,~]=eigs(round((Pi+Pi')/2,2));
beta_hat=U(:,1);

corr(beta_hat,beta)