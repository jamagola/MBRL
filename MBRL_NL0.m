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
% rf=readfis("fuzzyReward.fis")
LinearSystem = 1; %%%%%%%%%%%%%%%%%%  LINEAR EXPERIMENT %%%%%%%%%%%%%%%%%%%
% place '0' if non-linear MBRL is required

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
planSteps = 100;
defaultPenalty = 0;
model = cell(num_states, num_actions);

for s = 1:num_states
    for a = 1:num_actions
        model{s, a} = [s, defaultPenalty]; % initial next_state and reward
    end
end

% Initialize Q-values : Rough knowledge on action choices
%%%%%%%%%%%%%%%%%%%%%%%INITIALIZATION%%%%%%%%%%%%%%%%%%%%
Q = zeros(num_states, num_actions)-defaultPenalty; % using Q-table
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


% mode 0 LPM : mode 1 NLARX : mode 2 DynaQ
mode=2; %%%%%%%%%%%%%%%%%%%%%%%%%% IMPORTANT %%%%%%%%%%%%%%%%%%%%%%%
if mode == 0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% BUILD MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%
target_position = 0.0;  % Specific target position
i_=1;

if LinearSystem == 1
    F = 10; % Force : Change in System file too!
else
    F=1; % Force : Change in System file too!
end
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

%figure();
if LinearSystem == 1
    B_=[0];
else
    B_=[-10 -5 0 5 10];
end

for i=[1:length(B_)]
    x=0; %Initial Position
    v=0;
    B = B_(i);
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
        %subplot(3,2,i_);
        figure();
        compare(test_data,sysL1);
        if LinearSystem == 1
            title('Linear Model');
        else
            title(strcat("B=",string(B),"N"));
            i_=i_+1;
        end

    elseif i_==2
        sysL2=tfest(test_data, 2)
        sysL2D=c2d(sysL2,dt)
        N2=sysL2D.Numerator;
        D2=sysL2D.Denominator;
        %subplot(3,2,i_);
        figure();
        compare(test_data,sysL2);
        title(strcat("B=",string(B),"N"));
        i_=i_+1;
    elseif i_==3
        sysL3=tfest(test_data, 2)
        sysL3D=c2d(sysL3,dt)
        N3=sysL3D.Numerator;
        D3=sysL3D.Denominator;
        %subplot(3,2,i_);
        figure();
        compare(test_data,sysL3);
        title(strcat("B=",string(B),"N"));
        i_=i_+1;
    elseif i_==4
        sysL4=tfest(test_data, 2)
        sysL4D=c2d(sysL4,dt)
        N4=sysL4D.Numerator;
        D4=sysL4D.Denominator;
        %subplot(3,2,i_);
        figure();
        compare(test_data,sysL4);
        title(strcat("B=",string(B),"N"));
        i_=i_+1;
    elseif i_==5
        sysL5=tfest(test_data, 2)
        sysL5D=c2d(sysL5,dt)
        N5=sysL5D.Numerator;
        D5=sysL5D.Denominator;
        %subplot(3,2,i_);
        figure();
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
elseif mode == 1
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
else
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
end %%% END oF MODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L=0;
B=0;
F=10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if mode == 0

if LinearSystem == 1
    N_NL = N1;
    D_NL = D1;
else
    N_NL = [N1;N2;N3;N4;N5];
    D_NL = [D1;D2;D3;D4;D5];
end

% Dyna-Q learning
for episode = 1:num_episodes
    %%%%%%%%%%%%%%%%% IF YOU WANT RESET EVERY EPISODE
    u_1=0;
    u_2=0;
    %u_3=0; % for higher order
    
    y_1=0;
    y_2=0;
    %y_3=0; % for higher order
    %%%%%%%%%%%%%%%%% IF YOU WANT RESET EVERY EPISODE

    target_position=(-(range_/2)+rand()*(range_))/1; % randomly set %-2.5 to 2.5
    
    % if episode <= (num_episodes/3)
    %     target_position=(-(range_/2)+rand()*(range_/2))/1; % randomly set %-2.5 to 0
    % elseif episode <= 2*(num_episodes/3)
    %     target_position=(-(range_/4)+2*rand()*(range_/4))/1; % randomly set %-1.25 to 1.25
    % else
    %     target_position=(rand()*(range_/2))/1; % randomly set %0 to 2.5
    % end

    %target_position=1;
    x=0;
    %x=(-(range_/2)+rand()*(range_))/1; % randomly set %-2.5 to 2.5
    v=0;
    d2t = (x - target_position); %% d2tSign
    error = max(errorMin, min(errorMax,d2t)); % Clipped
    state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);

    for step_ = 1:max_steps
        % Epsilon-greedy action selection
        if rand() < epsilon
            action = randi(num_actions); % Explore
        else
            [~, action] = max(Q(state, :)); % Exploit
        end
        
        % Define action effects on the system
        if action == 1  % Push left
            force = -F;
        elseif action == 2  % 
            force = -F/2;
        elseif action == 3  % No force
            force = 0;
        elseif action == 4  % 
            force = F/2;
        elseif action == 5  % Push right
            force = F;
        end
        % No episode fail/termination
        % Simulate the environment (mass-spring-damper system)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [next_state, d2t, x,y_1,y_2,u_1,u_2]=workOrder2NL(num_states, target_position, N_NL,D_NL,force,u_1,u_2,y_1,y_2, LinearSystem);
        
        position(step_,1) = x;
        % Update the reward based on the distance to the target position
        distance_to_target(step_,1) = abs(d2t);
        % range_ -> -range_
        if step_ > 1
            tempReward = -(abs(d2t) - distance_to_target(step_-1,1))/(range_);
        else
            tempReward = 0;
        end
        % Can we use fuzzy logic??
        %reward0=gain_*(1 / (1 + distance_to_target(step_,1))); %(1/(range_+1) -> 1)
        reward0=-distance_to_target(step_,1); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATTENTION!
        %reward=evalfis(rf,[tempReward,reward0]);
        reward = reward0+zelta*tempReward; % Higher reward if closer to the target
        %%ATTENTION%%%%%
        %reward = zelta*tempReward; % Higher reward if closer to the target
        %reward = 10*(-distance_to_target(step,1));
        %reward = (1 / (1 + distance_to_target(step,1)));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%ACTION TO REWARD%%%%%%%%%%%%%%%%
        PM0 = -abs((num_actions - ceil((state/num_states)*(num_actions)) + 1) - action );
        
        % Q-value update
        Q(state, action) = Q(state, action) + eta*PM0 + alpha * (rewardAmp*reward + gamma * max(Q(next_state, :)) - Q(state, action)); % why this works?
            
        % Transition to the next state
        state = next_state;
        % Check if the target position is reached and break if so
        % if distance_to_target < 0.01  % Adjust the threshold 
        %     disp('Target achieved')
        %     break;
        % end
    end

    QQ(:,:,episode)=Q;
    waitbar(episode/num_episodes, f, sprintf('Progress: %d %%', floor((episode/num_episodes)*100)));
end

elseif mode == 1

% Dyna-Q learning
for episode = 1:num_episodes
    %%%%%%%%%%%%%%%%% IF YOU WANT RESET EVERY EPISODE
    u_=zeros(1,np);
    y_=zeros(1,np);
    past_data = iddata(y_', u_', dt);
    simOpt = simOptions('InitialCondition', past_data);
    %%%%%%%%%%%%%%%%% IF YOU WANT RESET EVERY EPISODE

    target_position=(-(range_/2)+rand()*(range_))/1; % randomly set %-2.5 to 2.5
    % if episode <= (num_episodes/3)
    %     target_position=(-(range_/2)+rand()*(range_/2))/1; % randomly set %-2.5 to 0
    % elseif episode <= 2*(num_episodes/3)
    %     target_position=(-(range_/4)+2*rand()*(range_/4))/1; % randomly set %-1.25 to 1.25
    % else
    %     target_position=(rand()*(range_/2))/1; % randomly set %0 to 2.5
    % end

    x=0;
    v=0;
    d2t = (x - target_position); %% d2tSign
    error = max(errorMin, min(errorMax,d2t)); % Clipped
    state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);

    for step_ = 1:max_steps
        % Epsilon-greedy action selection
        if rand() < epsilon
            action = randi(num_actions); % Explore
        else
            [~, action] = max(Q(state, :)); % Exploit
        end
        
        % Define action effects on the system
        if action == 1  % Push left
            force = -F;
        elseif action == 2  % 
            force = -F/2;
        elseif action == 3  % No force
            force = 0;
        elseif action == 4  % 
            force = F/2;
        elseif action == 5  % Push right
            force = F;
        end
        % No episode fail/termination
        % Simulate the environment (mass-spring-damper system)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        x = sim(sysNL, force, simOpt);
        u_=circshift(u_,-1);
        u_(end) = force; % Adding input
        y_=circshift(y_,-1);
        y_(end) = x;
        past_data = iddata(y_', u_', dt);
        simOpt = simOptions('InitialCondition', past_data);
        %[~, x, v, ~] = simulate_system(state, action, v, x, target_position);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Clip position to valid range [-range_/2, range_/2] Symmetry
        x_ = max(-range_/2, min(range_/2, x));
        x=x_;
        d2t = -(target_position - x);
        error = max(errorMin, min(errorMax,d2t)); % Clipped
        % Convert position and velocity back to the discrete state space
        % next_state = round(((x_ *(num_states-1))/range_+1));
        next_state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);
        % Define the reward function (e.g., maximize position)

        position(step_,1) = x;
        % Update the reward based on the distance to the target position
        distance_to_target(step_,1) = abs(d2t);
        % range_ -> -range_
        if step_ > 1
            tempReward = -(abs(d2t) - distance_to_target(step_-1,1))/(range_);
        else
            tempReward = 0;
        end
        % Can we use fuzzy logic??
        reward0=gain_*(1 / (1 + distance_to_target(step_,1))); %(1/(range_+1) -> 1)
        %reward0=-distance_to_target(step_,1); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATTENTION!
        %reward=evalfis(rf,[tempReward,reward0]);
        reward = reward0+zelta*tempReward; % Higher reward if closer to the target
        %%ATTENTION%%%%%
        %reward = zelta*tempReward; % Higher reward if closer to the target
        %reward = 10*(-distance_to_target(step,1));
        %reward = (1 / (1 + distance_to_target(step,1)));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%ACTION TO REWARD%%%%%%%%%%%%%%%%
        PM1 = -abs((num_actions - ceil((state/num_states)*(num_actions)) + 1) - action );
        
        % Q-value update
        Q(state, action) = Q(state, action) + eta*PM1 + alpha * (rewardAmp*reward + gamma * max(Q(next_state, :)) - Q(state, action)); % why this works?
            
        % Transition to the next state
        state = next_state;
        % Check if the target position is reached and break if so
        % if distance_to_target < 0.01  % Adjust the threshold 
        %     disp('Target achieved')
        %     break;
        % end
    end

    QQ(:,:,episode)=Q;
    waitbar(episode/num_episodes, f, sprintf('Progress: %d %%', floor((episode/num_episodes)*100)));
end

else
F=10;
B=0;
L=0;
simulate_system = @(state, action, v, x, target_position) simulate_mass_spring_damper(state, action, num_states, m, k, c, v, x, target_position, F, L, B);
% Dyna-Q learning
for episode = 1:num_episodes

    target_position=(-(range_/2)+rand()*(range_))/1; % randomly set %-2.5 to 2.5
    % if episode <= (num_episodes/3)
    %     target_position=(-(range_/2)+rand()*(range_/2))/1; % randomly set %-2.5 to 0
    % elseif episode <= 2*(num_episodes/3)
    %     target_position=(-(range_/4)+2*rand()*(range_/4))/1; % randomly set %-1.25 to 1.25
    % else
    %     target_position=(rand()*(range_/2))/1; % randomly set %0 to 2.5
    % end

    x=0;
    v=0;
    d2t = (x - target_position); %% d2tSign
    error = max(errorMin, min(errorMax,d2t)); % Clipped
    state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);

    for step_ = 1:max_steps
        % Epsilon-greedy action selection
        if rand() < epsilon
            action = randi(num_actions); % Explore
        else
            [~, action] = max(Q(state, :)); % Exploit
        end
        
        % Define action effects on the system
        if action == 1  % Push left
            force = -F;
        elseif action == 2  % 
            force = -F/2;
        elseif action == 3  % No force
            force = 0;
        elseif action == 4  % 
            force = F/2;
        elseif action == 5  % Push right
            force = F;
        end
        % No episode fail/termination
        % Simulate the environment (mass-spring-damper system)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [~, x, v, ~] = simulate_system(state, action, v, x, target_position);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Clip position to valid range [-range_/2, range_/2] Symmetry
        x_ = max(-range_/2, min(range_/2, x));
        x=x_;
        d2t = -(target_position - x);
        error = max(errorMin, min(errorMax,d2t)); % Clipped
        % Convert position and velocity back to the discrete state space
        % next_state = round(((x_ *(num_states-1))/range_+1));
        next_state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);
        % Define the reward function (e.g., maximize position)

        position(step_,1) = x;
        % Update the reward based on the distance to the target position
        distance_to_target(step_,1) = abs(d2t);
        % range_ -> -range_
        if step_ > 1
            tempReward = -(abs(d2t) - distance_to_target(step_-1,1))/(range_);
        else
            tempReward = 0;
        end
        % Can we use fuzzy logic??
        reward0=gain_*(1 / (1 + distance_to_target(step_,1))); %(1/(range_+1) -> 1)
        %reward0=-distance_to_target(step_,1); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATTENTION!
        %reward=evalfis(rf,[tempReward,reward0]);
        reward = reward0+zelta*tempReward; % Higher reward if closer to the target
        %%ATTENTION%%%%%
        %reward = zelta*tempReward; % Higher reward if closer to the target
        %reward = 10*(-distance_to_target(step,1));
        %reward = (1 / (1 + distance_to_target(step,1)));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%ACTION TO REWARD%%%%%%%%%%%%%%%%
        PM1 = -abs((num_actions - ceil((state/num_states)*(num_actions)) + 1) - action );
        
        % Q-value update
        Q(state, action) = Q(state, action) + eta*PM1 + alpha * (rewardAmp*reward + gamma * max(Q(next_state, :)) - Q(state, action)); % why this works?
        
        % Store in model and do 'planning'
        model{state,action} = [next_state, reward];
        
        for plan = 1:planSteps
            % Random synthetic experience
            s = randi([1,num_states]);
            a = randi([1,num_actions]);
            PM1_ = -abs((num_actions - ceil((s/num_states)*(num_actions)) + 1) - a);
            bundle = model{s,a};
            s_next=bundle(1);
            r = bundle(2);
            Q(s,a) = Q(s,a) + eta*PM1_ + alpha * (rewardAmp*r + gamma * max(Q(s_next,:)) - Q(s,a));
        end

        % Transition to the next state
        state = next_state;
        % Check if the target position is reached and break if so
        % if distance_to_target < 0.01  % Adjust the threshold 
        %     disp('Target achieved')
        %     break;
        % end
    end

    QQ(:,:,episode)=Q;
    waitbar(episode/num_episodes, f, sprintf('Progress: %d %%', floor((episode/num_episodes)*100)));
end

end

% final episodes
figure(); plot(position,'DisplayName','Position');hold;plot(distance_to_target,'r','DisplayName','Distance to Target');
hold on;
legend

[xx,yy]=meshgrid([1:num_actions],[1:num_states]);

if mode==1
    save nlarxmbrl.mat
else
    save nlmbrl.mat
end