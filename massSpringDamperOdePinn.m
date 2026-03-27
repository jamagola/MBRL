% massSpringDamperOdePinn.m
%
% Solve a ODE with a PINN model
% ODE is m*ddy + c*dy + k*y= f(t)
% IC's: y(0) = 0.0 and dy(0) = 0
% Custom loss function penalizing deviations from satisfying the ODE and
% the initial condictions:
% L = |m*ddy + c*dy + k*y - f(t)| + icCeoff|y(0)-0| + icCeoff|dy(0)-0| 
% _______________________________________________________________________
clear;clc
% Define time range and input data points
x = [0:0.01:5]'; % input data points in the range of 0 - 5

% Define the NN for solving the ODE. The input size is 1 (dimensions).
inputSize = 1;
layers = [
    featureInputLayer(inputSize, Normalization="none")
    fullyConnectedLayer(10)
    sigmoidLayer
    fullyConnectedLayer(1)
    sigmoidLayer];

% Create a DL network
net = dlnetwork(layers);

% Specify training parameters
numEpochs = 100;
miniBatchSize = 100;

% SGDM optimization, learning rate = 0.5, learning rate drop factor of 0.5,
% learning rate drop period of 5, momentum of 0.9.
initialLearnRate = 0.5;
learnRateDropFactor = 0.5;
learnRateDropPeriod = 5;
momentum = 0.9;

% Physical parameters for the MSD system
m = 1.0;  % Mass
c = 0.1;  % Damping coefficient
k = 2.0;  % Spring constant
x0 = 0.0; % Initial displacement
v0 = 0.0; % Initial velocity
wn = (k/m)^0.5;
zi = c/(2*m*wn);
icCoeff = 5; % Weight for initial condition loss

% External force function (e.g., sinusoidal force)
F = sin(x); % Force evaluated at time points x

% Create a datastore for training data
ads = arrayDatastore(x, IterationDimension=1); % Datastore object for training data 
mbq = minibatchqueue(ads, ...
    MiniBatchSize=miniBatchSize, ...
    PartialMiniBatch="discard", ...
    MiniBatchFormat="BC");

% Initialize solver (SGDM)
velocity = []; 

% Compute total number of iterations for training progress monitor
numObservationsTrain = numel(x);
numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

% Train the network using a custom training loop
monitor = trainingProgressMonitor( ...
    Metrics="LogLoss", ...
    Info=["Epoch" "LearnRate"], ...
    XLabel="Iteration");

epoch = 0;
iteration = 0;
learnRate = initialLearnRate;
start = tic;
t0=dlarray(0, "CB"); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Loop over epochs
while epoch < numEpochs
    epoch = epoch + 1;

    % Shuffle data
    mbq.shuffle;

    % Loop over mini-batches
    while hasdata(mbq)
        iteration = iteration + 1;
    
        % Read mini-batch of data
        X = next(mbq);
    
        % Evaluate the model gradients and loss using dlfeval and the MSD loss function
        [loss, gradients] = dlfeval(@modelLossMSD, net, X, F, m, c, k, x0, v0, t0, icCoeff); %%%%%%%%%%%%%%%%%%%%%%%%%%%
        %[loss, gradients] = dlfeval(@modelLossMSD, net, X, F, m, c, k, x0, icCoeff);
    
        % Update network parameters using the SGDM optimizer
        [net, velocity] = sgdmupdate(net, gradients, velocity, learnRate, momentum);
    
        % Update the training progress monitor
        recordMetrics(monitor, iteration, LogLoss=log(loss));
        updateInfo(monitor, Epoch=epoch, LearnRate=learnRate);
        monitor.Progress = 100 * iteration / numIterations;
    end

    % Reduce the learning rate
    if mod(epoch, learnRateDropPeriod) == 0
        learnRate = learnRate * learnRateDropFactor;
    end
end

% Testing of trained model
xTest = [0:0.01:4]';
yModel = minibatchpredict(net, xTest);

% Analytic solution (for comparison)
pp = 1/k*((k-m*wn^2)^2+c^2*wn^2)^0.5;
phi = atan(c*wn/(k-m*wn^2));
yAnalytic = x0.*cos(wn.*xTest)+pp.*cos(wn.*xTest-phi);
% Plot results
figure;
plot(xTest, yAnalytic, "-");
hold on;
plot(xTest, yModel, "--");
legend("Analytic", "PINN Model");
xlabel("x");
ylabel("y (log scale)");