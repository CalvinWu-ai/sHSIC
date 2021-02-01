function ynew=NWe(x,y,xnew)
% Nadaraya–Watson estimator with the Gaussian kernel 


options = optimset('Display','notify','TolX', 0.1);
h = fminsearch(@(h)BW_CV(x,y,h),1, options); % Leave-one-out CV tuning the bandwidth

temp=((xnew-x')./h).^2;
temp=exp(-temp/2); % Gaussian kernel
weight=temp./(sum(temp,2));
ynew=weight*y;

end


function cv=BW_CV(x,y,h)

temp=((x-x')./h).^2;
temp=exp(-temp/2); % Gaussian kernel
weight=temp./(sum(temp,2));
yhat=weight*y;
cv=mean(((y-yhat)./(1-diag(weight))).^2);

end


