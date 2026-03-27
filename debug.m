clear all; close all; clc;
% Build a dummy system
n=[1];
d=[1, 0.4, 1];
G=tf(n,d);
figure();
step(G);

steps=1000;
tf=1000;
%u=2*(double(randn(1,steps)>-1)-0.5);
%u=ones(1,steps);

fs = 1;                  % Sampling frequency (Hz)
T_ = 1000;                     % Total time duration (seconds)
numSwitches = 100;           % Number of random switches
minDuration = 1;          % Minimum duration of each state (seconds)
maxDuration = 100; 
u=pseudoIn(fs, T_, numSwitches, minDuration, maxDuration);

% Generate PRBS signal
%u = idinput(steps, 'prbs', [0 1], [0 1]);  % Generates a PRBS signal switching between -1 and 1
%u=u';
t=linspace(0,tf,steps);
y=lsim(G,u,t);

figure()
subplot(2,1,1);
plot(t,y,'r-', 'LineWidth', 2);
grid on;
xlabel('time');
ylabel('Response');

subplot(2,1,2);
plot(t,u,'k-', 'LineWidth', 1);
grid on;
xlabel('time');
ylabel('Input');


inputs=tonndata(u', false, false);
targets=tonndata(y, false, false);
lag=4;
hidden=[10];
in_delays=[1:lag];
feed_delays=[1:lag];
hiddenSize=[hidden,hidden];

narx_net = narxnet(in_delays, feed_delays, hiddenSize);

narx_net.divideFcn = 'divideblock';
narx_net.trainFcn = 'trainlm';

[X,Xi,Ai,T]=preparets(narx_net, inputs, {}, targets);

narx_net=train(narx_net, X,T,Xi,Ai);
view(narx_net);

%[net, tr]=nlarxWork(u,y');

u1=zeros(1,lag);
y1=zeros(1,lag);
Xi=num2cell([u1;y1]);
Xs=num2cell([u1(end);y1(end)]);

y_=lsim(G,ones(1,steps),t);
y_=[0;y_];

for i=1:400
    [Y,Xi,~]=narx_net(Xs,Xi,{});
    u1=[u1,1]; % Adding input
    y1=[y1,cell2mat(Y)];
    Xs=num2cell([u1(end);y1(end)]);
    %Xs=num2cell([u1(end);y_(i)]);
end

% Debug plot
figure();
plot(y1(1:100));