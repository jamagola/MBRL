%function [loss, gradients] = modelLossMSD(net, X, F, m, c, k, x0, icCoeff)
function [loss, gradients] = modelLossMSD(net, X, F, m, c, k, x0, v0, t0, icCoeff) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loss function for the mass-spring-damper (MSD) system
    % Inputs:
    %   net: Neural network model
    %   X: Input time points (dlarray)
    %   F: External force function evaluated at X
    %   m, c, k: Mass, damping coefficient, and spring constant
    %   x0, v0: Initial displacement and velocity
    %   icCoeff: Weight for initial condition loss

    % Forward pass to compute displacement y
    y = forward(net, X);

    % Compute first derivative of y with respect to time
    dy = dlgradient(sum(y, "all"), X, EnableHigherDerivatives=true);

    % Compute second derivative of y with respect to time
    ddy = dlgradient(sum(dy, "all"), X, EnableHigherDerivatives=true);

    % Define ODE residual (physics-informed loss)
    residual = (m * ddy + c * dy + k * y - F); 

    % Define initial condition losses
    ic_displacement = forward(net, dlarray(0, "CB")) - x0; % x(0) = x0 
    %ic_velocity = dlgradient(sum(forward(net, dlarray(0, "CB")), dlarray(0, "CB")) - v0; % x'(0) = v0
    ic_velocity = dlgradient(forward(net, dlarray(0, "CB")), t0) - v0; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

    % Combine losses: ODE residual and initial conditions
    ode_loss = mean(residual.^2, "all");
    ic_loss = icCoeff * (ic_displacement.^2 + ic_velocity.^2);
    %ic_loss = icCoeff * (ic_displacement.^2);

    % Total loss
    loss = ode_loss + ic_loss;

    % Compute gradients of the loss with respect to network parameters
    gradients = dlgradient(loss, net.Learnables);
end