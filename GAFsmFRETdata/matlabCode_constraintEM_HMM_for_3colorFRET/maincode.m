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

% list of data files to train on
filenames = {'../expData_3colorFRET/expCondition_461/group1/hel1_trace_4.dat', '../expData_3colorFRET/expCondition_461/group1/hel2_trace_3.dat'};

% read all traces into cell arrays
ydataAll = cell(1, length(filenames));
timeAll  = cell(1, length(filenames));
for f = 1:length(filenames)
    data = readmatrix(filenames{f});
    t = data(:,1)';
    t = t - t(1);
    timeAll{f} = t;
    yd = data(:,2:4)';
    yd(2,:) = data(:,4)'; yd(3,:) = data(:,3)';
    % now, the 1 row of yd is green, 2 row is blue, and 3 row is red!
    % Note: per-trace max-normalization was intentionally removed. The raw
    % data is already background-subtracted (zero-FRET baseline anchored
    % near 0 across traces), so dividing each trace by its own max would
    % introduce inconsistency between traces that visit different states.
    ydataAll{f} = yd;
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% expected states
% first row is green, second row is blue, third row is red
% estimation of FRET and covariance for each color fluorescence
% Initial high/mid/low intensity levels per color, in raw count units.
% These replaced the previous [0,1] FRET-scale guesses (e.g. ug=[0.90 0.48 0.00])
% after per-trace max-normalization was removed: the data now enters the EM in
% raw counts, so the initial means and sigmas must be on the same scale.
% Values derived from the typical condition-wide raw maxima (~430 green,
% ~170 blue, ~310 red) scaled by the original FRET ratios.
ug = [430 230 0];   % green
ub = [120  56 5];   % blue
ur = [250 110 3];   % red
sigmag = [70, 60, 40];
sigmab = [25, 30, 18];
sigmar = [40, 55, 30];

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

[u, sigma2, A, ProbMax, PI] = expectationMaximization_Algorithm(stateindex, u, sigma2, PI, A, ydataAll);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot figures
% plot the likehood
plot(ProbMax,'-b','linewidth',2)
xlabel('EM Iterations');
ylabel('Likelihood');
set(gca,'linewidth', 2,'fontsize',20,'fontname','Times New Roman');
set(gcf,'unit','centimeters','position',[10 6 20 13]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Viterbi and output per trace
baseDir = pwd;
for f = 1:length(filenames)
    ydata = ydataAll{f};
    time  = timeAll{f};
    filename = filenames{f};

    % Vertibi to solve the state sequence
    [stateSequenceFinal, probMax, probSequenceFinal] = stateSequence_Viterbi_Algorithm(ydata, u, sigma2, A, PI);

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
    ylabel('Fitted intensity');
    set(gca,'linewidth', 2,'fontsize',20,'fontname','Times New Roman');

    % state sequence
    ax3 = nexttile;
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
    xlabel(t,'Time Step','fontsize',20,'fontname','Times New Roman')
    xticklabels(ax1,{})
    xticklabels(ax2,{})
    t.TileSpacing = 'compact';
    t.Padding = 'compact';
    set(gcf,'unit','centimeters','position',[10 6 40 30]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % output data into files
    foldername = filename(1:(length(filename)-4));
    if exist(foldername, 'dir'),
	    rmdir(foldername, 's');
    end
    mkdir(foldername)
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

    cd(baseDir)
end
