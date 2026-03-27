% Using model-based RL to control a mass spring damper system
% Actions are pushing left, applying no force, or pushing right
% Model is in discrete time
% September 4th, 2023, MPS

% Note
% simulate_mass_spring_damper.m
% pseudoIn.m
% members_.m
% members_u.m
% animate_.m
% validate.m
% workOrder2NL.m
% fuzzyReward.fis
% must be in common workspace

% Main Non-linear MBRL Research Script

clear;clc;close all;
format long
rf=readfis("fuzzyReward.fis")

% Parameters for the mass-spring-damper system
m = 1;          % Mass (kg)
k = 4;         % Spring constant (N/m)
c = 1;          % Damping coefficient (Ns/m)
f = waitbar(0, 'Starting');
% Discretize the state space and action space

range_=5;
num_states = 100;    % Number of discrete states
num_actions = 5;     % Number of discrete actions (push left, no force, right)
num_episodes = 300;  % Number of episodes
alpha = 0.5;         % Learning rate
gamma = 0.9;         % Discount factor
epsilon = 0.1;       % Exploration rate
rewardAmp=2; %reward amplification
max_steps = 3000;     % Maximum number of steps per episode
position = zeros(max_steps,1);   % Position
distance_to_target = zeros(max_steps,1);
target_position = 0.0;  % Specific target position
F=10; % Force : Change in SYtem file too!
B=0;
%act=zeros(max_steps,1);
%V=zeros(max_steps,1);
tempReward=0;
dt=0.01; %sampling rate
np=2; % Order
gain_=range_;
zelta = 1;
eta = 20.0;
good=1;
% Initialize Q-values : Rough knowledge on action choices
%%%%%%%%%%%%%%%%%%%%%%%INITIALIZATION%%%%%%%%%%%%%%%%%%%%
Q = zeros(num_states, num_actions); % using Q-table
% for s=1:num_states
%     for a=1:num_actions
%         if(ceil(s/(num_states/num_actions))==a)
%             Q(s,a)=good;
%         end
%     end
% end
% Q=fliplr(Q);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
QQ= zeros(num_states, num_actions, num_episodes);

errorMax=1*range_;
errorMin=-1*range_;

% Function to simulate the mass-spring-damper system
L=0; % Mode - hard boundary %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
simulate_system = @(state, action, v, x, target_position) simulate_mass_spring_damper(state, action, num_states, m, k, c, v, x, target_position, F, L, B);

x=0; %Initial Position
v=0;

d2t = (x - target_position); %%d2tSign
error = max(errorMin, min(errorMax,d2t)); % Clipped
state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);



mode=0; %%%%%%%%%%%%%%%%%%%%%%%%%% IMPORTANT %%%%%%%%%%%%%%%%%%%%%%%
if mode == 0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% BUILD MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%
target_position = 0.0;  % Specific target position
i_=1;
F=1; % Force : Change in System file too!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt=0.01; %sampling rate
% Initialize Q-values
errorMax=1*range_;
errorMin=-1*range_;

% Function to simulate the mass-spring-damper system
simulate_system = @(state, action, v, x, target_position) simulate_mass_spring_damper(state, action, num_states, m, k, c, v, x, target_position, F, L, B);

steps=12000;
tf=steps*dt;

fs = 1/dt;                  % Sampling frequency (Hz)
T_ = tf;                    % Total time duration (seconds)
numSwitches = 100;          % Number of random switches
minDuration = 500*dt;      % Minimum duration of each state (seconds)
maxDuration = 500*dt; 

figure();

for B=[-10 -5 0 5 10]

    x=0; %Initial Position
    v=0;
    
    d2t = (x - target_position); %% d2tSign
    error = max(errorMin, min(errorMax,d2t)); % Clipped 
    state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);
    simulate_system = @(state, action, v, x, target_position) simulate_mass_spring_damper(state, action, num_states, m, k, c, v, x, target_position, F, L, B);
    %%%%%%%%%%%%%%
    action=pseudoIn(fs, T_, numSwitches, minDuration, maxDuration);
    u=[action==5]*F+[action==1]*-F+[action==3]*0+[action==4]*(F/2)+[action==2]*(-F/2);
    u=u+B;
    %%%%%%%%%%%%%%
    t = 0:1/fs:T_-1/fs;
    y=zeros(size(u));
    for j=1:steps
        [next_state, x, v, d2t] = simulate_system(state, action(j), v, x, target_position);
        state=next_state;
        y(j)=x;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    test_data = iddata(y',u',dt);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

    if i_==1
        sysL1=tfest(test_data, 2)
        sysL1D=c2d(sysL1,dt)
        N1=sysL1D.Numerator;
        D1=sysL1D.Denominator;
        subplot(3,2,i_);
        %figure();
        compare(test_data,sysL1);
        title(strcat("B=",string(B),"N"));
        i_=i_+1;
    elseif i_==2
        sysL2=tfest(test_data, 2)
        sysL2D=c2d(sysL2,dt)
        N2=sysL2D.Numerator;
        D2=sysL2D.Denominator;
        subplot(3,2,i_);
        %figure();
        compare(test_data,sysL2);
        title(strcat("B=",string(B),"N"));
        i_=i_+1;
    elseif i_==3
        sysL3=tfest(test_data, 2)
        sysL3D=c2d(sysL3,dt)
        N3=sysL3D.Numerator;
        D3=sysL3D.Denominator;
        subplot(3,2,i_);
        %figure();
        compare(test_data,sysL3);
        title(strcat("B=",string(B),"N"));
        i_=i_+1;
    elseif i_==4
        sysL4=tfest(test_data, 2)
        sysL4D=c2d(sysL4,dt)
        N4=sysL4D.Numerator;
        D4=sysL4D.Denominator;
        subplot(3,2,i_);
        %figure();
        compare(test_data,sysL4);
        title(strcat("B=",string(B),"N"));
        i_=i_+1;
    elseif i_==5
        sysL5=tfest(test_data, 2)
        sysL5D=c2d(sysL5,dt)
        N5=sysL5D.Numerator;
        D5=sysL5D.Denominator;
        subplot(3,2,i_);
        %figure();
        compare(test_data,sysL5);
        title(strcat("B=",string(B),"N"));
        i_=i_+1;
    else
        % Nothing yet!
    end
end

F=10;
B=0;
L=0;
simulate_system = @(state, action, v, x, target_position) simulate_mass_spring_damper(state, action, num_states, m, k, c, v, x, target_position, F, L, B);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%% BUILD MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%
steps=12000;
tf=steps*dt;

fs = 1/dt;                  % Sampling frequency (Hz)
T_ = tf;                     % Total time duration (seconds)
numSwitches = 100;           % Number of random switches
minDuration = 500*dt;        % Minimum duration of each state (seconds)
maxDuration = 500*dt; 
action=pseudoIn(fs, T_, numSwitches, minDuration, maxDuration);
u=[action==5]*F+[action==1]*-F+[action==3]*0+[action==4]*(F/2)+[action==2]*(-F/2);
t = 0:1/fs:T_-1/fs;
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

lag=np;
test_data = iddata(y',u',dt);

%sysNL = nlarx(test_data,[5 5 1], idSigmoidNetwork('NumberOfUnits', [10, 10]))
opt = nlarxOptions;
opt.Regularization.Lambda = 1e-8; % Adjust λ based on data
sysNL = nlarx(test_data, [2 2 1], 'idSigmoidNetwork', opt);
figure()
compare(test_data,sysNL)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
end %%% END oF MODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%