% Function to simulate the mass-spring-damper system dynamics - Used for
% the Journal
function [next_state, x, v, d2t] = simulate_mass_spring_damper(state, action, num_states, m, k, c, v, x, target_position,F, L, B)
    % Convert state to continuous variables (position and velocity)
    range_=5; % -2.5 to 2.5
    errorMax=1*range_;
    errorMin=-1*range_;
    %x = range_*(state - 1) / (num_states - 1);
    
    if ((x <=-range_/2) || (x >=range_/2)) && (L==0) 
        v = 0; % V always zero? No
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    d2t = -(target_position - x);
    error = max(errorMin, min(errorMax,d2t)); % Clipped
    %F=10*abs(error);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    
    %%%%%%%%%%%%%%%
    force=force+B;
    %%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%

    % Update the system dynamics using a simple Euler integration
    dt = 0.01;  % Time step
    hardening=1;
    g=1;
    acceleration = force / m - (k / m)*sin(g*x) - (c / m) * v - (hardening/m)*(x.^3); % NON LINEAR !!!
    %acceleration = force / m - (k / m) * x - (c / m) * v;
    v = v + acceleration * dt;
    x = x + v * dt;
    
    if L==0 
        % Clip position to valid range [-range_/2, range_/2] Symmetry
        x_ = max(-range_/2, min(range_/2, x));
        x=x_;
        d2t = -(target_position - x); %% d2tSign
        error = max(errorMin, min(errorMax,d2t)); % Clipped
        % Convert position and velocity back to the discrete state space
        % next_state = round(((x_ *(num_states-1))/range_+1));
        next_state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);
        % Define the reward function (e.g., maximize position)
    else % pretty dumb
        % Clip position to valid range [-range_/2, range_/2] Symmetry
        temp = x;
        x_ = max(-range_/2, min(range_/2, x));
        x=x_;
        d2t = -(target_position - x); %%% d2tSign
        error = max(errorMin, min(errorMax,d2t)); % Clipped
        % Convert position and velocity back to the discrete state space
        % next_state = round(((x_ *(num_states-1))/range_+1));
        next_state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);
        % Define the reward function (e.g., maximize position)
        x=temp;
    end
end