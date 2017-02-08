% RuKu integrator for FitzHugh system with parameters 
% Modified from Voss et al 2004.
function r=vossFNfct(dq,x)
global dT dt nn
p=x(1:dq,:); 
xnl=x(dq+1:size(x(:,1)),:); 
for n=1:nn
  k1=dt*fc(xnl,p);
  k2=dt*fc(xnl+k1/2,p);
  k3=dt*fc(xnl+k2/2,p);
  k4=dt*fc(xnl+k3,p);
  xnl=xnl+k1/6+k2/3+k3/3+k4/6;
end
r=[p; xnl];
end

function r=fc(x,params);
p = params(1,:);
a = params(2,:);
b = params(3,:);
% a = .7;
% b=.8; 
c=3.;
r=[c.*(x(2,:)+x(1,:)-x(1,:).^3/3+p); -(x(1,:)-a+b.*x(2,:))./c];
end