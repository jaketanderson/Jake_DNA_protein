%
stateNum = 6;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read the experimental data
% Note: the experimental data: time: 1st column.
%                              green: 2nd column
%                              red: 3rd column
%                              blue: 4th column 
% NOTE!: the paper shows blue before red color!

% read the data
filename = 'hel3_trace_9.dat';
data = readmatrix(filename);
time = data(:,1)'; % time step is 0.05 s 
time = time - time(1);
ydata = data(:,2:4)'; 
ydata(2,:) = data(:,4)'; ydata(3,:) = data(:,3)'; 
% now, the 1 row of ydata is green, 2 row is blue, and 3 row is red!

% normalize 
ydata(1,:) = ydata(1,:) / max(ydata(1,:));
ydata(2,:) = ydata(2,:) / max(ydata(2,:));
ydata(3,:) = ydata(3,:) / max(ydata(3,:));
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% expected states
% first row is green, second row is blue, third row is red
% estimation of FRET and covariance for each color fluorescence
ug = [0.90 0.48 0.00]; % FRET of green
ub = [0.70 0.33 0.03]; % FRET of blue
ur = [0.80 0.35 0.01]; % FRET of red
sigmag = [0.15, 0.12, 0.08];
sigmab = [0.15, 0.18, 0.10];
sigmar = [0.12, 0.17, 0.10];

% build the system states according to the couplings between states
stateindex = [3 1 2 2 2 1  
              3 3 1 2 3 3
              3 3 3 2 1 3]; 
u = zeros(3,stateNum);
u = build_states_from_stateindex(stateindex, ug, ub, ur);

sigma2 = zeros(3,3,stateNum);
for i = 1:stateNum  
    a1 = sigmag(stateindex(1,i));
    a2 = sigmab(stateindex(2,i));
    a3 = sigmar(stateindex(3,i));
    sigma2(:,:,i) = [a1^2, 0, 0
                     0, a2^2, 0
                     0, 0, a3^2];
end

% transition matrix. Estimated by counting the transitons of the experimental trajectory. hel3_trace_17.dat
A =[0.9396, 0.0001, 0.0108, 0.0495, 0.0001, 0.0001
    0.0472, 0.5279, 0.4121, 0.0127, 0,      0
    0.0147, 0.0597, 0.8055, 0.1005, 0.3194, 0
    0.0001, 0.0555, 0.3654, 0.2535, 0.3252, 0.0005
    0.0001, 0,      0.3418, 0.2869, 0.5712, 0.0001
    0.0001, 0,      0,      0.0001, 0.0001, 0.9999
];
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Expection-Maximization Algorithm
PI  = ones(1,stateNum) / stateNum; % imission prob at t=0

[u, sigma2, A, ProbMax, PI] = expectationMaximization_Algorithm(stateindex, u, sigma2, PI, A, ydata);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vertibi to sovle the state sequence
[stateSequenceFinal, probMax, probSequenceFinal] = stateSequence_Viterbi_Algorithm(ydata, u, sigma2, A, PI);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot figures
% plot the likehood 
plot(ProbMax,'-b','linewidth',2)
xlabel('EM Iterations');
ylabel('Likelihood');
ylim([0,1]);
set(gca,'linewidth', 2,'fontsize',20,'fontname','Times New Roman');
set(gcf,'unit','centimeters','position',[10 6 20 13]); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot the fitted curve
figure
t = tiledlayout(3,1);

% original data
ax1 = nexttile;
plot(time, ydata(1,:),'-go','markersize',5,'linewidth',2)
hold on
plot(time, ydata(2,:),'-bs','markersize',5,'linewidth',2)
plot(time, ydata(3,:),'-rs','markersize',5,'linewidth',2)
ylabel('FRET');
%ylim([-0.2,1]);
yticks(-0.2:0.2:1);
set(gca,'linewidth', 2,'fontsize',20,'fontname','Times New Roman');

% fitted FRET 
ax2 = nexttile;
yfit = zeros(3, size(ydata,2));
for i = 1:size(ydata,2)
    stateIndex = stateSequenceFinal(i);
    yfit(1,i) = u(1,stateIndex);
    yfit(2,i) = u(2,stateIndex);
    yfit(3,i) = u(3,stateIndex);
end
plot(time, yfit(1,:),'g','linewidth',2)
hold on
plot(time, yfit(2,:),'b','linewidth',2)
plot(time, yfit(3,:),'r','linewidth',2)
ylabel('Fitted-FRET');
ylim([-0.2,1]);
yticks(-0.2:0.2:1);
set(gca,'linewidth', 2,'fontsize',20,'fontname','Times New Roman');

% state sequence
ax3 = nexttile;
% add dots for state index, just for clear show of each state point
newStateSequence = [];
newProbSequence = [];
newTime = [];
for i = 1:length(time)
    stateIndex = stateSequenceFinal(i);
    tmpState = [stateIndex, stateIndex, stateIndex];
    newStateSequence = [newStateSequence, tmpState];
    
    prob = probSequenceFinal(i);
    tmpProb = [prob, prob, prob];
    newProbSequence = [newProbSequence, tmpProb];
    
    tmpTime = [time(i)-0.05/3, time(i), time(i)+0.05/3]; % 0.05s is the original time step
    newTime = [newTime, tmpTime];
end
newStateSequence = newStateSequence - 1;
plot(newTime,newStateSequence,'-k','linewidth',2)

ylabel('State Index');
ylim([-0.5,5.5]);
yticks(0:1:5);
set(gca,'linewidth', 2,'fontsize',20,'fontname','Times New Roman');

linkaxes([ax1,ax2,ax3],'x');
%xlim([0,10]);
xlabel(t,'Time Step','fontsize',20,'fontname','Times New Roman')
xticklabels(ax1,{})
xticklabels(ax2,{})
t.TileSpacing = 'compact';
t.Padding = 'compact';
set(gcf,'unit','centimeters','position',[10 6 40 30]); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% output data into files 
% the x-axis, is time, time step is 0.05s 
%mkdir outputData
%cd("outputData")
foldername = filename(1:(length(filename)-4));
mkdir (foldername)
cd(foldername)

originalData = [time; ydata];
writematrix(originalData,'originalData.txt')

fittedData = [time; yfit];
writematrix(fittedData,'fittedData.txt')

stateSequence = [newTime; newStateSequence];
writematrix(stateSequence,'stateSequence.txt')
writematrix(newProbSequence,'probSequence.txt')

writematrix(u,'systemStateFRET.txt')

writematrix(sigma2,'systemStateCovariance.txt')

writematrix(A,'transitionProbability.txt')

cd ..


