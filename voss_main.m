clear all; close all;
% Algorighm originally made available by Henning U. Voss, 2002 "Nonlinear dynamical system identification
% from uncertain and indirect measurements"
% HU Voss, J Timmer, J Kurths - International Journal
% of Bifurcation and Chaos, Vol. 14, No. 6 (2004) 1905-1933 Reproduces Figure 9 from paper
% Modifications by S. Schiff 2009 - Matlab and Octave Compatible
global dT dt nn % Sampling time step as global variable 
dq=3; dx=dq+2; dy=1;
% Dimensions: (dq for param. vector, dx augmented state, dy observation) 
fct='vossFNfct'; % this is the model function F(x) used in filtering 
obsfct='vossFNobsfct'; % this is the observation function G(x)
N=800; % number of data samples
dT=0.2; % sampling time step (global variable)
dt=dT; nn=fix(dT/dt); % the integration time step can be smaller than dT 

% Preallocate arrays
x0=zeros(2,N); % Preallocates x0, the underlying true trajectory 
xhat=zeros(dx,N); % Preallocate estimated x
Pxx=zeros(dx,dx,N); % Prallocate Covariance in x
errors=zeros(dx,N); % Preallocate errors
Ks=zeros(dx,dy,N); % Preallocate Kalman gains 

% Initial Conditions
x0(:,1)=[0; 0]; % initial value for x0

% External input current, estimated as parameter p later on:
params = zeros(dq,N);
params(1,:) = (1:N)/250*2*pi; 
params(1,:) =-.4-1.01*abs(sin(params(1,:)/2));
params(2,:) = .7; % a
params(3,:) = .8; % b

% RuKu integrator of 4th order:
for n=1:N-1;
    xx=x0(:,n); 
    for i=1:nn
        k1=dt*vossFNint(xx,params(:,n));
        k2=dt*vossFNint(xx+k1/2,params(:,n));
        k3=dt*vossFNint(xx+k2/2,params(:,n));
        k4=dt*vossFNint(xx+k3,params(:,n));
        xx=xx+k1/6+k2/3+k3/3+k4/6;
    end;
    x0(:,n+1)=xx;
end;
x=[params; x0]; % augmented state vector (notation a bit different to paper) 
xhat(:,1)=x(:,1); % first guess of x_1 set to observation

% Covariances
Q=diag([.15,.015, .015]); % process noise covariance matrix 
R=.2^2*var(vossFNobsfct(x))*eye(dy,dy); % observation noise covariance matrix 
% randn('state',0);
y=feval(obsfct,x)+sqrtm(R)*randn(dy,N); % noisy data
Pxx(:,:,1)=blkdiag(Q,R,R);% Initial Condition for Pxx 
% Main loop for recursive estimation
for k=2:N
    [xhat(:,k),Pxx(:,:,k),Ks(:,:,k)]=...
        vossut(xhat(:,k-1),Pxx(:,:,k-1),y(:,k),fct,obsfct,dq,dx,dy,R);
    Pxx(1:dq,1:dq,k)=Q; 
    errors(:,k)=sqrt(diag(Pxx(:,:,k))); 
end; % k
% Results 
chisq=...
    mean((x(1,:)-xhat(1,:)).^2+(x(2,:)-xhat(2,:)).^2+(x(3,:)-xhat(3,:)).^2) 
est=xhat(1:dq,N)'; % last estimate
error=errors(1:dq,N)'; % last error
meanest=mean(xhat(1:dq,:),2)
meanerror=mean(errors(1:dq,:)') % Plot Results
subplot(2,1,1)
plot(y,'bd','MarkerEdgeColor','blue', 'MarkerFaceColor','blue',... 
    'MarkerSize',3);
hold on; plot(x(dq+1,:),'k','LineWidth',2); xlabel('t');
ylabel('x_1, y');
hold off;
axis tight
title('(a)')
subplot(2,1,2)
plot(x(dq+2,:),'k','LineWidth',2);
hold on
plot(xhat(dq+2,:),'r','LineWidth',2); 

plot(x(1,:),'k','LineWidth',2);
for i=1:dq; plot(xhat(i,:),'m','LineWidth',2); end; 
for i=1:dq; plot(xhat(i,:)+errors(i,:),'m'); end; 
for i=1:dq; plot(xhat(i,:)-errors(i,:),'m'); end; 
xlabel('t');
ylabel('z, estimated z, x_2, estimated x_2');
hold off
axis tight 
title('(b)')

