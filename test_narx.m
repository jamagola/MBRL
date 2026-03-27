% Using model-based RL to control a mass spring damper system
% Actions are pushing left, applying no force, or pushing right
% Model is in discrete time
% September 4th, 2023, MPS

% Note
% simulate_mass_spring_damper.m must be in common workspace

clear;clc;close all;
format long
rf=readfis("fuzzyReward.fis")

% Parameters for the mass-spring-damper system
m = 1;          % Mass (kg)
k = 4;         % Spring constant (N/m)
c = 1;          % Damping coefficient (Ns/m)
v = 0;
f = waitbar(0, 'Starting');
% Discretize the state space and action space

range_=2;
num_states = 100;    % Number of discrete states
num_actions = 3;     % Number of discrete actions (push left, no force, right)
num_episodes = 100;  % Number of episodes
alpha = 0.1;         % Learning rate
gamma = 0.9;         % Discount factor
epsilon = 0.1;       % Exploration rate
max_steps = 3000;     % Maximum number of steps per episode
position = zeros(max_steps,1);   % Position
distance_to_target = zeros(max_steps,1); % #########################
target_position = 0.3;  % Specific target position
F=10; % Force : Change in SYtem file too!
%act=zeros(max_steps,1);
%V=zeros(max_steps,1);
tempReward=0;
dt=0.01; %sampling rate
np=4; % Order
gain_=2;
% Initialize Q-values
%Q = zeros(num_states, num_actions); % using Q-table
QQ= zeros(num_states, num_actions, num_episodes);
Q = zeros(num_states, num_actions); %%%%%%%%%%%%%%
errorMax=1*range_;
errorMin=-1*range_;

rewardAmp=2; %reward amplification
% or randi
% Initialize model of the environment
%model = struct('next_state', ones(num_states, num_actions), 'reward', zeros(num_states, num_actions));

% Function to simulate the mass-spring-damper system
L=0; % Mode
simulate_system = @(state, action, v, x, target_position) simulate_mass_spring_damper(state, action, num_states, m, k, c, v, x, target_position, F, L);

u_=zeros(1,np);
y_=zeros(1,np);

x=0; %Initial Position
v=0;

d2t = (x - target_position);
error = max(errorMin, min(errorMax,d2t)); % Clipped
state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% BUILD MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%
steps=12000;
tf=steps*dt;

fs = 1/dt;                  % Sampling frequency (Hz)
T_ = tf;                     % Total time duration (seconds)
numSwitches = 100;           % Number of random switches
minDuration = 1000*dt;          % Minimum duration of each state (seconds)
maxDuration = 1000*dt; 
action=pseudoIn(fs, T_, numSwitches, minDuration, maxDuration);
u=[action==3]*F+[action==1]*-F+[action==2]*0;
%u=[action==3]*1+[action==1]*-1;
t=linspace(0,tf,steps);
y=zeros(size(u));
for j=1:steps
    [next_state, x, v, d2t] = simulate_system(state, action(j), v, x, target_position);
    state=next_state;
    y(j)=x;
end
dispF= steps;
figure()
subplot(2,1,1);
plot(t(1:dispF),y(1:dispF),'r-', 'LineWidth', 2);
grid on;
xlabel('time');
ylabel('Response');

subplot(2,1,2);
plot(t(1:dispF),u(1:dispF),'b-', 'LineWidth', 2);
grid on;
xlabel('time');
ylabel('Input');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Test and debug
u=u';
y=y';

test_data = iddata(y,u,dt);

%sysNL = nlarx(test_data,[5 5 1], idSigmoidNetwork('NumberOfUnits', [10, 10]))
sysNL = nlarx(test_data,[5 5 1])
figure()
compare(test_data,sysNL)