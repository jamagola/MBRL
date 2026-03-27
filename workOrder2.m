function[next_state, d2t, x,y_1,y_2,u_1,u_2]=workOrder2(num_states, target_position, N,D,force,u_1,u_2,y_1,y_2)
% used in the Journal
    range_=2;
    errorMax=1*range_;
    errorMin=-1*range_;
    
    
    
    U=[force;u_1;u_2];
    Y=[y_1;y_2];
    %x=N*U-D(2:end)*Y;
    x=N(2:end)*U(1:end-1)-D(2:end)*Y;
   
    
    % Clip position to valid range [-range_/2, range_/2] Symmetry
    x_ = max(-range_/2, min(range_/2, x));
    x=x_;
    d2t = -(target_position - x);
    error = max(errorMin, min(errorMax,d2t)); % Clipped
    % Convert position and velocity back to the discrete state space
    % next_state = round(((x_ *(num_states-1))/range_+1));
    next_state = round(((num_states-1)/(errorMax-errorMin))*(error-errorMin) + 1);
    % Define the reward function (e.g., maximize position)

    u_2=u_1;
    u_1=force;
    y_2=y_1;
    y_1=x;

end