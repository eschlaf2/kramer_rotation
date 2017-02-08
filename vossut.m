% Unscented transformation from Voss et al 2004
% This Function has been modified by S. Schiff and T. Sauer 2008 
function [xhat,Pxx,K]=vossut(xhat,Pxx,y,fct,obsfct,dq,dx,dy,R)
N=2*dx; %Number of Sigma Points
Pxx=(Pxx+Pxx')/2; %Symmetrize Pxx - good numerical safety
xsigma=chol(dx*Pxx)'; % Cholesky decomposition - note that Pxx=chol'*chol
Xa=xhat*ones(1,N)+[xsigma, -xsigma]; %Generate Sigma Points 
X=feval(fct,dq,Xa); %Calculate all of the X's at once
xtilde = mean(X,2);
% xtilde=sum(X')'/N; %Mean of X's 
X1=X-xtilde*ones(1,size(X,2)); % subtract mean from X columns 
Pxx=X1*X1'/N;
Pxx=(Pxx+Pxx')/2; %Pxx covariance calculation
Y=feval(obsfct,X);
ytilde=mean(Y,2); %sum(Y')'/N;
Y1=Y-ytilde*ones(1,size(Y,2)); % subtract mean from Y columns 
Pyy=Y1*Y1'/N + R; %Pyy covariance calculation
Pxy=X1*Y1'/N; %cross-covariance calculation
K=Pxy/Pyy; 
xhat=xtilde+K*(y-ytilde); 
Pxx=Pxx-K*Pxy'; Pxx=(Pxx+Pxx')/2;