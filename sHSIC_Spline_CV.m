function [Pi,optlambda,cv,cv_error]=sHSIC_Spline_CV(x,y,lambdas,nfold,seed,initPi)

[n,~]=size(x);
rng(seed);
index=randperm(n);
%fold=repmat(1:nfold,1,(length(y)-1)/nfold)';% 5 fold cross validation
%fold=[fold;5];
fold=repmat(1:nfold,1,length(y)/nfold)';% 5 fold cross validation
fold=fold(index);
cv_error=zeros(nfold,length(lambdas));
quiet=1;

for i=1:nfold
    ytrain=y(fold~=i);
    ytest=y(fold==i);
    xtrain=x(fold~=i,:);
    xtest=x(fold==i,:);
    Pi=initPi;
    if quiet>0
        fprintf('cross validation for dataset %2d\n ',i);
    end
    for j=1:length(lambdas)
        if quiet>1
            fprintf('%2d th lambda begins\n ',j);
        end
        Pi=sHSIC(xtrain,ytrain,lambdas(j),struct('verbosity',0,'initPi',Pi,'outer_tol',1e-6,'outer_maxiter',10000)); 
        [U,~]=eigs((Pi+Pi')/2);
        xtemp=xtrain*U(:,1);
        xnew=xtest*U(:,1);
        yhat=NWe(xtemp,ytrain,xnew);
        cv_error(i,j)=mean((ytest-yhat).^2);           
    end    
end
cv=mean(cv_error,1);
[~,index]=min(cv);
optlambda=lambdas(index);
Pi=sHSIC(x,y,optlambda,struct('verbosity',0,'initPi',Pi,'outer_tol',1e-6,'outer_maxiter',10000)); 
end