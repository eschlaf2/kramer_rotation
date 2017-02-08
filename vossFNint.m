%This function calculates the Fitzhugh-Nagumo equations 
function r=vossFNint(x,params)
z = params(1);
a = params(2);
b = params(3);
% b=.8; 
c=3.;
r=[c.*(x(2)+x(1)-x(1)^3/3+z); -(x(1)-a+b.*x(2))./c];