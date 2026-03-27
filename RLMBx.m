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
k = 10;         % Spring constant (N/m)
c = 1;          % Damping coefficient (Ns/m)
v = 0;
f = waitbar(0, 'Starting');
% Discretize the state space and action space
modelBased=0.01; % How much of the episodes for model learning
range_=2;
num_states = 100;    % Number of discrete states
num_actions = 3;     % Number of discrete actions (push left, no force, right)
num_episodes = 1000;  % Number of episodes
alpha = 0.1;         % Learning rate
gamma = 0.9;         % Discount factor
epsilon = 0.1;       % Exploration rate
max_steps = 3000;     % Maximum number of steps per episode
position = zeros(max_steps,1);   % Position
distance_to_target = zeros(max_steps,1); % #########################
target_position = 0.3;  % Specific target position
x=0; %Initial Position
act=zeros(max_steps,1);
V=zeros(max_steps,1);
tempReward=0;
dt=0.01; %sampling rate
np=4; % Order
last=50; % last n experiments for the model estimation
N=0;
D=0;
% Initialize Q-values
Q = zeros(num_states, num_actions); % using Q-table
QQ= zeros(num_states, num_actions, num_episodes);

errorMax=1*range_;
errorMin=-1*range_;

rewardAmp=2; %reward amplification
% or randi
% Initialize model of the environment
%model = struct('next_state', ones(num_states, num_actions), 'reward', zeros(num_states, num_actions));

% Function to simulate the mass-spring-damper system
simulate_system = @(state, action, v, x, target_position) simulate_mass_spring_damper(state, action, num_states, m, k, c, v, x, target_position);



u_1=0;
u_2=0;
u_3=0;
u_4=0; % for higher order

y_1=0;
y_2=0;
y_3=0;
y_4=0; % for higher order

dt=0.01; %sampling rate

time_=(np-1)*dt;
t=[0:dt:(np-1)*dt];
u=[zeros(1,np)];
y=[zeros(1,np)];

x=0; %Initial Position
v=0;
index=1+np;

% Dyna-Q learning
for episode = 1:num_episodes
    u_1=0;
    u_2=0;
    u_3=0;
    u_4=0; % for higher order
    
    y_1=0;
    y_2=0;
    y_3=0;
    y_4=0; % for higher order

    target_position=-(range_/2)+rand()*range_; % randomly set %-1 to 1
    x=0;
    v=0;
    d2t = (x - target_position);
    error = max(errorMin, min(errorMax,d2t)); % Clipped
    state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);

    for step = 1:max_steps
        % Epsilon-greedy action selection
        if rand() < epsilon
            action = randi(num_actions); % Explore
        else
            [~, action] = max(Q(state, :)); % Exploit
        end
        
        F=10;
        % Define action effects on the system
        if action == 1  % Push left
            force = -F;
        elseif action == 2  % No force
            force = 0;
        elseif action == 3  % Push right
            force = F;
        end
        % No episode fail/termination
        % Simulate the environment (mass-spring-damper system)
        if episode <= round(num_episodes*modelBased)
            [next_state, x, v, d2t] = simulate_system(state, action, v, x, target_position);
            position(step,1) = x;
            V(step, 1)=v;
            act(step, 1)=action;
            % Update the reward based on the distance to the target position
            distance_to_target(step,1) = abs(d2t);
            
            if step > 1
                tempReward = -(abs(d2t) - distance_to_target(step-1,1))/(range_);
            else
                tempReward = 0;
            end
            % Can we use fuzzy logic??
            reward0=(1 / (1 + distance_to_target(step,1))); %(1/(range_+1) -> 1)
            %reward=evalfis(rf,[tempReward,reward0]);
            reward = reward0+tempReward; % Higher reward if closer to the target
            %reward = 10*(-distance_to_target(step,1));
            %reward = (1 / (1 + distance_to_target(step,1)));
        else

            range_=2;
            errorMax=1*range_;
            errorMin=-1*range_;

            %U=[u_2,u_1];
            U=[u_4, u_3, u_2, u_1, force];
            Y=[y_4, y_3, y_2, y_1];
            x = nlarx_(0,0,U,Y);
            
            % Clip position to valid range [-range_/2, range_/2] Symmetry
            x_ = max(-range_/2, min(range_/2, x));
            x=x_;
            d2t = -(target_position - x);
            error = max(errorMin, min(errorMax,d2t)); % Clipped
            % Convert position and velocity back to the discrete state space
            % next_state = round(((x_ *(num_states-1))/range_+1));
            next_state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);
            % Define the reward function (e.g., maximize position)
            
            u_4=u_3;
            u_3=u_2;
            u_2=u_1;
            u_1=force;
            y_4=y_3;
            y_3=y_2;
            y_2=y_1;
            y_1=x;

            position(step,1) = x;
            % Update the reward based on the distance to the target position
            distance_to_target(step,1) = abs(d2t);
            % range_ -> -range_
            if step > 1
                tempReward = -(abs(d2t) - distance_to_target(step-1,1))/(range_);
            else
                tempReward = 0;
            end
            % Can we use fuzzy logic??
            reward0=(1 / (1 + distance_to_target(step,1))); %(1/(range_+1) -> 1)
            %reward=evalfis(rf,[tempReward,reward0]);
            reward = reward0+tempReward; % Higher reward if closer to the target
            %reward = 10*(-distance_to_target(step,1));
            %reward = (1 / (1 + distance_to_target(step,1)));
        end

        % Q-value update
        Q(state, action) = Q(state, action) + alpha * (rewardAmp*reward + gamma * max(Q(next_state, :)) - Q(state, action)); % why this works?
        
        % Model/data update
        if (episode <= round(num_episodes*modelBased)) && (step ~= max_steps) 
            time_=time_+dt;
            u(index)=force;
            y(index)=x;
            t(index)=time_;
            index=index+1;
        elseif (episode <= round(num_episodes*modelBased)) && (step == max_steps) 
            time_=time_+dt;
            u(index)=force;
            y(index)=x;
            t(index)=time_;

            u(index)
            index=index+1;

            time_=(np-1)*dt;
t=[0:dt:(np-1)*dt];
u=[zeros(1,np)];
y=[zeros(1,np)];

        end 
        
        % Transition to the next state
        state = next_state;
        % Check if the target position is reached and break if so
        % if distance_to_target < 0.01  % Adjust the threshold 
        %     disp('Target achieved')
        %     break;
        % end
    end

    % Build model (tfest,NARX,...) np=order
    if episode == round(num_episodes*modelBased)
        [net,tr]=nlarxWork(u(end-6000:end),y(end-6000:end))
        %[net,tr]=nlarxWork(u,y)
        disp("Model build done:");
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        u_1=u(end);
        u_2=u(end-1);
        u_3=u(end-2);
        u_4=u(end-3);
        y_1=y(end);
        y_2=y(end-1);
        y_3=y(end-2);
        y_4=y(end-3);
    end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    QQ(:,:,episode)=Q;
    waitbar(episode/num_episodes, f, sprintf('Progress: %d %%', floor((episode/num_episodes)*100)));
end

% final episodes
figure(1); plot(position,'DisplayName','Position');hold;plot(distance_to_target,'r','DisplayName','Distance to Target');
hold on;
legend
%figure(2);
%compare(temp,Hd);

if modelBased == 1
    figure(3);
    plot(act,'g','DisplayName','Action');
    legend
    grid on;
    figure(4);
    plot(V);
    title('velocity');
end

[xx,yy]=meshgrid([1:num_actions],[1:num_states]);
