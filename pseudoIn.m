function[u] = pseudoIn(fs, T, numSwitches, minDuration, maxDuration)
%used in the Journal

t = 0:1/fs:T-1/fs;          % Time vector

% Generate random switching times
switchTimes = cumsum(minDuration + (maxDuration - minDuration) * rand(1, numSwitches));

% Ensure switchTimes do not exceed the total time T
switchTimes = switchTimes(switchTimes < T);

% Initialize the square wave signal
u = 2*ones(size(t));         % Initialize the input signal
cs=3;
% Create the square wave
for i = 1:length(switchTimes)
    % Set the signal to the current state until the next switch
    
    switch cs
        case 2
            cs = randi(3);
        case 1
            cs = randi(2);
        case 4
            cs = randi(3)+2;
        case 5
            cs = randi(2)+3;
        otherwise
            cs = randi(3)+1;
    end
    u(t >= switchTimes(i)) = cs;
end
