% Using model-based RL to control a mass spring damper system
% Actions are pushing left, applying no force, or pushing right
% Model is in discrete time

% Note
% simulate_mass_spring_damper.m must be in common workspace

clear;clc;close all;
format long

% Parameters for the mass-spring-damper system
m = 1;          % Mass (kg)
k = 4;         % Spring constant (N/m)
c = 1;          % Damping coefficient (Ns/m)

% Discretize the state space and action space

range_=5;
num_states = 100;    % Number of discrete states
num_actions = 5;     % Number of discrete actions (push left, no force, right)

target_position = 0.0;  % Specific target position
i_=1;
figure();
%for F=[0.1 0.5 1 2 5 10]
for B=[-10 -5 -1  1 5 10]

F=1; % Force : Change in SYtem file too!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt=0.01; %sampling rate
% Initialize Q-values
errorMax=1*range_;
errorMin=-1*range_;

% Function to simulate the mass-spring-damper system
L=0; % Mode
simulate_system = @(state, action, v, x, target_position) simulate_mass_spring_damper(state, action, num_states, m, k, c, v, x, target_position, F, L, B);

x=0; %Initial Position
v=0;

d2t = (x - target_position);
error = max(errorMin, min(errorMax,d2t)); % Clipped
state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% BUILD MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%
steps=12000;
tf=steps*dt;

fs = 1/dt;                  % Sampling frequency (Hz)
T_ = tf;                    % Total time duration (seconds)
numSwitches = 100;          % Number of random switches
minDuration = 500*dt;      % Minimum duration of each state (seconds)
maxDuration = 500*dt; 
action=pseudoIn(fs, T_, numSwitches, minDuration, maxDuration);
u=[action==5]*F+[action==1]*-F+[action==3]*0+[action==4]*(F/2)+[action==2]*(-F/2);
%%%%%%%%%%%%%%
u=u+B;

t=linspace(0,tf,steps);
y=zeros(size(u));
for j=1:steps
    [next_state, x, v, d2t] = simulate_system(state, action(j), v, x, target_position);
    state=next_state;
    y(j)=x;
end

% dispF= steps;
% figure()
% subplot(2,1,1);
% plot(t(1:dispF),y(1:dispF),'r-', 'LineWidth', 2);
% grid on;
% xlabel('time');
% ylabel('Response');
% 
% subplot(2,1,2);
% plot(t(1:dispF),u(1:dispF),'b-', 'LineWidth', 2);
% grid on;
% xlabel('time');
% ylabel('Input');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_data = iddata(y',u',dt);

%sysNL = nlarx(test_data,[5 5 1], idSigmoidNetwork('NumberOfUnits', [10, 10]))
%sysNL = nlarx(test_data,[5 5 1]) % Wavelet network

% opt = nlarxOptions;
% opt.Regularization.Lambda = 1e-8; % Adjust λ based on data
% sysNL = nlarx(test_data, [2 2 1], 'idSigmoidNetwork', opt);
% figure()
% compare(test_data,sysNL)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

if i_==1
    sysL1=tfest(test_data, 2)
    subplot(3,2,i_);
    compare(test_data,sysL1);
    title(strcat("B=",string(B),"N"));
    i_=i_+1;
elseif i_==2
    sysL2=tfest(test_data, 2)
    subplot(3,2,i_);
    compare(test_data,sysL2);
    title(strcat("B=",string(B),"N"));
    i_=i_+1;
elseif i_==3
    sysL3=tfest(test_data, 2)
    subplot(3,2,i_);
    compare(test_data,sysL3);
    title(strcat("B=",string(B),"N"));
    i_=i_+1;
elseif i_==4
    sysL4=tfest(test_data, 2)
    subplot(3,2,i_);
    compare(test_data,sysL4);
    title(strcat("B=",string(B),"N"));
    i_=i_+1;
elseif i_==5
    sysL5=tfest(test_data, 2)
    subplot(3,2,i_);
    compare(test_data,sysL5);
    title(strcat("B=",string(B),"N"));
    i_=i_+1;
else
    sysL6=tfest(test_data, 2)
    subplot(3,2,i_);
    compare(test_data,sysL6);
    title(strcat("B=",string(B),"N"));
    i_=i_+1;
end
%opt = nlarxOptions;
%opt.Regularization.Lambda = 1e-8; % Adjust λ based on data
%sysNL = nlarx(test_data, [2 2 1], 'idSigmoidNetwork', opt);

% subplot(3,2,i);
% compare(test_data,sysL);
% title(strcat("B=",string(B),"N"));
% i=i+1;
end