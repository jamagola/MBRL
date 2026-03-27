close all; clc; clear all;

% Parameters for the mass-spring-damper system: Check with simulate MSD
% file and the main MBRL routine for manual update!

m = 1;          % Mass (kg)
k = 4;         % Spring constant (N/m)
c = 1;          % Damping coefficient (Ns/m)
hardening=1;
g=1;

gain_ = 5;
maxErr = 5;
zelta = 1;

err = [0:0.5:maxErr];
deltaErr = [-maxErr:0.5:maxErr]; % current error - past error

Fitness = gain_*(1 ./ (1 + err));
%fitnessB = -err;
Penalty = -deltaErr/maxErr;

figure();

plot(deltaErr, Penalty, 'r--', 'LineWidth', 2);
hold on;
%plot(err, fitnessB, 'r--', 'LineWidth', 2);
%hold on;
plot(err, Fitness, 'g-', 'LineWidth', 2);
hold on;

xlabel('|Error| or Delta-Error');
ylabel('Fitness/Cost');
title('Reward-Penalty');
legend('Penalty','Fitness');
grid on;

r=1;
c=1;
for i=[1:length(err)]
    for j=[1:length(deltaErr)]
        if (err(i) < deltaErr(j)) 
            r=1;
            break;
        end
        if (deltaErr(j)<(err(i) - maxErr))
            continue;
        end
        errY(r,c)=err(i);
        deltaErrX(r,c)=deltaErr(j);
        r=r+1;
    end
    c=c+1;
end


rewardA = gain_*(1 ./ (1 + errY))-zelta*deltaErrX/maxErr;
%rewardB = -errY-zelta*deltaErrX/maxErr;
figure();
surf(deltaErrX, errY, rewardA);
xlabel('delta error');
ylabel('|error|');
zlabel('Reward');
title('Reward Function');

% figure();
% xlabel('delta error');
% ylabel('error');
% zlabel('Reward');
% title('RewardB');
% surf(deltaErrX, errY, rewardB);

p1 = linspace(-2,-1,10); 
frc1 = 12.77*p1;
frc1T = (k / m)*sin(g*p1) + (hardening/m)*(p1.^3);
err1= mean(abs(frc1T-frc1));

p2 = linspace(-1.5,-0.5,10); 
frc2 = 6.218*p2;
frc2T = (k / m)*sin(g*p2) + (hardening/m)*(p2.^3);
err2= mean(abs(frc2T-frc2));

p3 = linspace(-1,1,10); 
frc3 = 4.036*p3;
frc3T = (k / m)*sin(g*p3) + (hardening/m)*(p3.^3);
err3= mean(abs(frc3T-frc3));

p4 = linspace(0.5,1.5,10); 
frc4 = 5.458*p4;
frc4T = (k / m)*sin(g*p4) + (hardening/m)*(p4.^3);
err4= mean(abs(frc4T-frc4));

p5 = linspace(1,2,10); 
frc5 = 9.797*p5;
frc5T = (k / m)*sin(g*p5) + (hardening/m)*(p5.^3);
err5= mean(abs(frc5T-frc5));
MAE_k = mean([err1, err2, err3, err4, err5])

p = linspace(-2,2,100);
frc = (k / m)*sin(g*p) + (hardening/m)*(p.^3);
figure();
plot(p, frc, 'k', lineWidth=2);
xlabel('Position');
ylabel('Force');
title('Non-linear Spring Behavior');
hold on;
plot(p1, frc1, 'c--', LineWidth=2);
hold on;
plot(p2, frc2, 'm--', LineWidth=2);
hold on;
plot(p3, frc3, 'r--', LineWidth=2);
hold on;
plot(p4, frc4, 'g--', LineWidth=2);
hold on;
plot(p5, frc5, 'b--', LineWidth=2);
hold on;
legend("Non-linear", "Linear Model 1", "Linear Model 2", "Linear Model 3", "Linear Model 4", "Linear Model 5");