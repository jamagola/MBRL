% This script assumes these variables are defined:
%
%   u - input time series.
%   y - feedback time series.
function[net,tr]=nlarxWork(u,y)
X = tonndata(u,true,false);
T = tonndata(y,true,false);

trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.


% Create a Nonlinear Autoregressive Network with External Input
inputDelays = 1:4;
feedbackDelays = 1:4;
hiddenLayerSize = [10,10];
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);

%net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
%net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

% Prepare the Data for Training and Simulation
% The function PREPARETS prepares timeseries data for a particular network,
% shifting time by the minimum amount to fill input states and layer
% states. Using PREPARETS allows you to keep your original time series data
% unchanged, while easily customizing it for networks with differing
% numbers of delays, with open loop or closed loop feedback modes.
[x,xi,ai,t] = preparets(net,X,{},T);

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'time';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.performFcn = 'mse';  % Mean Squared Error

% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate', 'ploterrhist', ...
    'plotregression', 'plotresponse', 'ploterrcorr', 'plotinerrcorr'};

% Train the Network
[net,tr] = train(net,x,t,xi,ai);

%net = feedforwardnet(10);
%[net,tr] = train(net,x,t);

% View the Network
%view(net)

genFunction(net,'nlarx_','MatrixOnly','yes');

% x1 = cell2mat(x(1,:));
% x2 = cell2mat(x(2,:));
% xi1 = cell2mat(xi(1,:));
% xi2 = cell2mat(xi(2,:));
% 
% y = nlarx_(0,0,xi1,xi2);

end