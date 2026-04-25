% Train/test cross-validation across experimental conditions.
% For each condition: 80/20 random split (rng seed 123), train HMM on the
% 80% set, evaluate held-out log-likelihood on the 20% set, and run Viterbi
% on both sets. Outputs land in <baseOut>/<conditionName>/.

baseDir = pwd;

stateNum = 6;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% conditions to process
conditions = { ...
    '../expData_3colorFRET/expCondition_461/group1', ...
    '../expData_3colorFRET/expCondition_461/group2', ...
    '../expData_3colorFRET/expCondition_SHL7', ...
    '../expData_3colorFRET/expCondition_lowMg2'};

baseOut = 'crossval_output';
if ~exist(baseOut, 'dir'), mkdir(baseOut); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initial parameters (same starting point for each condition)
ug0 = [0.90 0.48 0.00];
ub0 = [0.70 0.33 0.03];
ur0 = [0.80 0.35 0.01];
sigmag = [0.15, 0.12, 0.08];
sigmab = [0.15, 0.18, 0.10];
sigmar = [0.12, 0.17, 0.10];

stateindex = [3 1 2 2 2 1
              3 3 1 2 3 3
              3 3 3 2 1 3];

A0 =[0.9396, 0.0001, 0.0108, 0.0495, 0.0001, 0.0001
     0.0472, 0.5279, 0.4121, 0.0127, 0,      0
     0.0147, 0.0597, 0.8055, 0.1005, 0.3194, 0
     0.0001, 0.0555, 0.3654, 0.2535, 0.3252, 0.0005
     0.0001, 0,      0.3418, 0.2869, 0.5712, 0.0001
     0.0001, 0,      0,      0.0001, 0.0001, 0.9999];

PI0 = ones(1,stateNum) / stateNum;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set rng once for reproducibility across all conditions
rng(1234);

summaryRows = {};
summaryRows{end+1} = sprintf('%-50s %8s %8s %14s %14s %14s %14s', ...
    'condition', 'nTrain', 'nTest', 'trainLL', 'testLL', 'trainLL/T', 'testLL/T');

for c = 1:length(conditions)
    condDir = conditions{c};
    files = dir(fullfile(condDir, '*.dat'));
    nFiles = length(files);
    if nFiles < 2
        fprintf('Skipping %s (only %d files)\n', condDir, nFiles);
        continue
    end

    % build full paths
    filenames = cell(1, nFiles);
    for k = 1:nFiles
        filenames{k} = fullfile(condDir, files(k).name);
    end

    % 80/20 split
    perm = randperm(nFiles);
    nTrain = max(1, round(0.8 * nFiles));
    trainIdx = perm(1:nTrain);
    testIdx  = perm(nTrain+1:end);

    % load all traces (normalize each by its own max, as in maincode.m)
    ydataAll = cell(1, nFiles);
    timeAll  = cell(1, nFiles);
    for k = 1:nFiles
        data = readmatrix(filenames{k});
        t = data(:,1)'; t = t - t(1);
        timeAll{k} = t;
        yd = data(:,2:4)';
        yd(2,:) = data(:,4)'; yd(3,:) = data(:,3)';
        yd(1,:) = yd(1,:) / max(yd(1,:));
        yd(2,:) = yd(2,:) / max(yd(2,:));
        yd(3,:) = yd(3,:) / max(yd(3,:));
        ydataAll{k} = yd;
    end

    ydataTrain = ydataAll(trainIdx);
    ydataTest  = ydataAll(testIdx);

    % initial parameters
    u0 = build_states_from_stateindex(stateindex, ug0, ub0, ur0);
    sigma0 = zeros(3,3,stateNum);
    for i = 1:stateNum
        a1 = sigmag(stateindex(1,i));
        a2 = sigmab(stateindex(2,i));
        a3 = sigmar(stateindex(3,i));
        sigma0(:,:,i) = diag([a1^2, a2^2, a3^2]);
    end

    % train
    fprintf('Training on %s (%d train, %d test)...\n', condDir, length(trainIdx), length(testIdx));
    [u, sigma2, A, ProbMax, PI] = expectationMaximization_Algorithm( ...
        stateindex, u0, sigma0, PI0, A0, ydataTrain);

    % validation log-likelihoods
    [trainLL, trainPerTrace, trainT] = computeLogLikelihood(u, sigma2, PI, A, ydataTrain);
    [testLL,  testPerTrace,  testT ] = computeLogLikelihood(u, sigma2, PI, A, ydataTest);

    % output directory for this condition
    [~, condName1] = fileparts(condDir);
    parentName = '';
    [parentDir, ~] = fileparts(condDir);
    [~, parentBase] = fileparts(parentDir);
    if startsWith(condName1, 'group')
        condTag = sprintf('%s_%s', parentBase, condName1);
    else
        condTag = condName1;
    end
    outDir = fullfile(baseOut, condTag);
    if exist(outDir, 'dir'), rmdir(outDir, 's'); end
    mkdir(outDir);

    % save trained parameters
    cd(outDir)
    writematrix(u,      'systemStateFRET.txt')
    writematrix(sigma2, 'systemStateCovariance.txt')
    writematrix(A,      'transitionProbability.txt')
    writematrix(PI,     'initialProb.txt')
    writematrix(ProbMax(:), 'emLikelihoodHistory.txt')

    % validation summary
    fid = fopen('validation_loglik.txt','w');
    fprintf(fid, 'condition: %s\n', condDir);
    fprintf(fid, 'nTrain=%d, nTest=%d\n', length(trainIdx), length(testIdx));
    fprintf(fid, 'trainTotalT=%d, testTotalT=%d\n', trainT, testT);
    fprintf(fid, 'trainLogLik=%.6f, testLogLik=%.6f\n', trainLL, testLL);
    fprintf(fid, 'trainLogLik/T=%.6f, testLogLik/T=%.6f\n', trainLL/trainT, testLL/testT);
    fprintf(fid, '\nper-train-trace logLik:\n');
    for k = 1:length(trainIdx)
        fprintf(fid, '  %s\t%.6f\n', filenames{trainIdx(k)}, trainPerTrace(k));
    end
    fprintf(fid, '\nper-test-trace logLik:\n');
    for k = 1:length(testIdx)
        fprintf(fid, '  %s\t%.6f\n', filenames{testIdx(k)}, testPerTrace(k));
    end
    fclose(fid);

    % Viterbi on each trace, into train/ and test/ subfolders
    mkdir('train'); mkdir('test');
    runViterbiSet('train', filenames(trainIdx), ydataAll(trainIdx), timeAll(trainIdx), u, sigma2, A, PI);
    runViterbiSet('test',  filenames(testIdx),  ydataAll(testIdx),  timeAll(testIdx),  u, sigma2, A, PI);

    cd(baseDir)

    summaryRows{end+1} = sprintf('%-50s %8d %8d %14.4f %14.4f %14.6f %14.6f', ...
        condTag, length(trainIdx), length(testIdx), trainLL, testLL, ...
        trainLL/trainT, testLL/testT);
end

% write summary across conditions
summaryFile = fullfile(baseOut, 'summary.txt');
fid = fopen(summaryFile, 'w');
for k = 1:length(summaryRows)
    fprintf(fid, '%s\n', summaryRows{k});
end
fclose(fid);

fprintf('\nDone. Summary written to %s\n', summaryFile);
for k = 1:length(summaryRows)
    fprintf('%s\n', summaryRows{k});
end


function runViterbiSet(subdir, fnames, yset, tset, u, sigma2, A, PI)
    setBase = pwd;
    cd(subdir)
    for k = 1:length(fnames)
        ydata = yset{k};
        time  = tset{k};
        [stateSeq, ~, probSeq] = stateSequence_Viterbi_Algorithm(ydata, u, sigma2, A, PI);

        yfit = zeros(3, size(ydata,2));
        for i = 1:size(ydata,2)
            s = stateSeq(i);
            yfit(:,i) = u(:,s);
        end

        [~, traceName] = fileparts(fnames{k});
        if exist(traceName, 'dir'), rmdir(traceName, 's'); end
        mkdir(traceName)
        cd(traceName)
        writematrix([time; ydata], 'originalData.txt')
        writematrix([time; yfit],  'fittedData.txt')
        writematrix([time; stateSeq-1], 'stateSequence.txt')
        writematrix(probSeq, 'probSequence.txt')
        cd ..
    end
    cd(setBase)
end
