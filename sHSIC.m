function Pi=sHSIC(x,y,lambda,option)

% [Pi,object]=sHSIC(x,y,0.01)
% lambda: Penalty coefficient

[n,p]=size(x);
covx=cov(x);
[eigvec,eigval]=eig((covx+covx')/2);
sqcovx=real(eigvec*sqrt(max(eigval,0))*eigvec');
sigmaY2=var(y);
rho=1;% LADMM penalty parameter
tau=2*rho*eigs(covx,1)^2; % Linearized coefficient

%------------------------------------------------defaults for option
outer_maxiter=5000;
outer_tol=1e-6;
inner_maxiter=1000;
inner_tol=1e-3;
verbosity=2; % verbosity>=2 display every iteration message, otherwise display the final convergent message
initPi=zeros(p);
%-------------------------------------------------

% User's option
%------------------------------------------------
if nargin > 3  %  then paramstruct is an argument
    if isfield(option,'outer_maxiter')
        outer_maxiter = option.outer_maxiter; 
    end
    if isfield(option,'outer_tol')
        outer_tol = option.outer_tol; 
    end 
    if isfield(option,'inner_maxiter')
        inner_maxiter = option.inner_maxiter; 
    end
    if isfield(option,'inner_tol')
        inner_tol = option.inner_tol; 
    end 
    if isfield(option,'verbosity')
        verbosity = option.verbosity;
    end    
    if isfield(option,'initPi')
        initPi = option.initPi;
    end
end
%---------------------------------------------------


sy=diag(y*y');
Ly=sy+sy'-2*(y*y');
Ly=exp(-Ly./(2*sigmaY2));
Ly=Ly-mean(Ly,2)-mean(Ly,1)+mean(mean(Ly)); % normalization

Coef=-Ly.*(Ly<0)/4;
Coef=x'*(2*(diag(sum(Coef,2))-Coef)/(n^2))*x;
L=eigs((Coef+Coef')/2,1,'la'); % Hessian Upper Bound Approximation

if verbosity >= 2
    fprintf('   iter   sub_iter    cost val \t  \t \t      tol  \n');
end

% main code
Pi=initPi;
H=sqcovx*Pi*sqcovx;
Gamma=zeros(p);
outer_error=1;
for outer_iter=1:outer_maxiter
    
    if outer_error < outer_tol
        if verbosity>0
            fprintf('sHSIC terminates: converged iteration:%4d\n', outer_iter);
        end
        break;
    end
    
      
    %Gradient computation at the current point
    sx=diag(x*Pi*x');
    Kx=sx+sx'-2*(x*Pi*x');
    Kx=exp(-Kx./2);
    Coeft=(Kx.*Ly)./2;
    dF=x'*(2*(diag(sum(Coeft,2))-Coeft)/(n^2))*x;
    
    a=Pi-dF/L;  
    Pi_pre=Pi;
    inner_error=1;
    inner_iter=0;
    %-----------------LADMM inner cycle-----------------------
    while inner_iter<=inner_maxiter && inner_error> inner_tol
        
        temp=Pi-rho/tau*covx*Pi*covx+rho/tau*sqcovx*(H-Gamma)*sqcovx;
        temp=tau/(tau+L)*temp+L/(tau+L)*a;
        Pi=max(abs(temp)-lambda/(L+tau),0).*sign(temp); % Soft-thresholding operator
        
        H=FantopeProjection(sqcovx*Pi*sqcovx+Gamma);
        Gamma=Gamma+sqcovx*Pi*sqcovx-H;	       
        
        inner_error=norm(sqcovx*Pi*sqcovx-H,'fro');
        inner_iter=inner_iter+1;  
    end
    %---------------------------------------------------------
        
    outer_error=norm(Pi-Pi_pre,'fro');
    
    %Compute the objective value
    sx=diag(x*Pi*x');
    Kx=sx+sx'-2*(x*Pi*x');
    Kx=exp(-Kx./2);
    F_trial=mean(mean(Kx.*Ly))-lambda*sum(sum(abs(Pi)));
 
    % Display iteration information
    if verbosity >= 2
        fprintf('%5d  \t %5d \t  %.8e  \t      %.8e  \n', ...
                outer_iter, inner_iter, F_trial, outer_error);
    end    
    
    if outer_iter == outer_maxiter
        if verbosity>0
            disp('sHSIC terminates: Achieved maximum iteration.');
        end
        break;
    end
    
end
end


function H=FantopeProjection(W)

% This code is to solve the projection problem onto the fantope constraint
%   min_H || H-W ||
%     s.t. ||H||_{*}<=K, ||H||_{sp}<=1 
temp=(W+W')/2;
[V, D] = eig(temp);
d = diag(D);

if sum(min(1,max(d,0)))<= 1
    gamma=0;    
else
    knots=unique([(d-1);d]);
    knots=sort(knots,'descend');
    temp=find(sum(min(1,max(d-knots',0)))<=1);
    lentemp=temp(end);
    a=knots(lentemp);
    b=knots(lentemp+1);
    fa=sum(min(1,max(d-a,0)));
    fb=sum(min(1,max(d-b,0)));
    gamma=a+(b-a)*(1-fa)/(fb-fa);  
end

d_final=min(1,max(d-gamma,0));
H = V * diag(d_final) * V';


end
