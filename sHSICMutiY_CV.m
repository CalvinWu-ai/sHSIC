function [Pi,optlambda,cv,cv_error]=sHSICMutiY_CV(x,y,lambdas,nfold,seed,initPi)

[n,q]=size(y);
rng(seed);
index=randperm(n);
fold=repmat(1:nfold,1,length(y)/nfold)';% 5 fold cross validation
fold=fold(index);
cv_error=zeros(length(nfold),length(lambdas));
quiet=1;

for i=1:nfold
    ytrain=y(fold~=i,:);
    ytest=y(fold==i,:);
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
        Pi=sHSICMutiY(xtrain,ytrain,lambdas(j),struct('verbosity',0,'initPi',Pi)); 
        [U,~]=eigs((Pi+Pi')/2);
        xtemp=xtrain*U(:,1);
        xnew=xtest*U(:,1);
        yhat=zeros(size(ytest));
        for k=1:q
            yhat(:,k)=NWe(xtemp,ytrain(:,k),xnew);            
        end
        cv_error(i,j)=mean(sum((ytest-yhat).^2,2));           
    end    
end
cv=mean(cv_error,1);
[~,index]=min(cv);
optlambda=lambdas(index);
Pi=sHSICMutiY(x,y,optlambda,struct('verbosity',0,'initPi',Pi)); 
end