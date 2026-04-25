
function[U_op, Sigma_op, A_op, likeliHood, PI_op] = expectationMaximization_Algorithm(stateindex, u, sigma, PI, A, ydataAll)
%%%%%%%%%%
% function for computing posterior probabilities of HMM using EM algorithm
% impose gaussian probabilities on emission probabilities,
% parametirized by u and covariance matrix sigma.

% Math is based on the book:
% Bishop- Pattern Recognition and Machine Learning - Springer (2006)
% Chapter 13

% observed data: y(t).  states: q(k).
% ydataAll: cell array of traces, each trace is d x T_n

% iterate the algorithm for convergence

likeliHood = [];

sigma0 = sigma;

nTraces = length(ydataAll);
d = size(ydataAll{1}, 1);
Nstate = size(u,2);

[Mg, Mb, Mr] = build_selectMatrix(stateindex);

criterion = 1e-3;
isConverged = false;
iter = 100;
tooManyIteration = false;
time = 0;
while isConverged == false && tooManyIteration == false

    time = time + 1;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Accumulate sufficient statistics across all traces
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % accumulators for M-step
    PI_acc      = zeros(Nstate, 1);
    A_num       = zeros(Nstate, Nstate);
    A_den       = zeros(Nstate, 1);
    gamaSumT    = zeros(Nstate, 1);   % sum_n sum_t gamma_n(i,t)
    gamaSumY    = zeros(d, Nstate);   % sum_n y_n * gamma_n'
    gammaSumOld = zeros(Nstate, 1);   % for covariance update using uold
    covAcc      = zeros(d, d, Nstate);
    totalLogLik = 0;

    uold = u;

    for n = 1:nTraces
        y = ydataAll{n};
        T = size(y, 2);

        pyq   = zeros(Nstate, T);
        alpha = zeros(Nstate, T);
        beta  = zeros(Nstate, T);
        gamma = zeros(Nstate, T);
        xce   = zeros(Nstate, Nstate, T-1);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % forward, alpha
        pyq(:,1) = gaussmmC(u, sigma, y(:,1), d);
        for i = 1:Nstate
            alpha(i,1) = PI(i) * pyq(i,1);
        end
        alphasum = sum(alpha(:,1));
        alpha(:,1) = alpha(:,1)/alphasum;

        for t = 1:T-1
            pyq(:,t+1) = gaussmmC(u, sigma, y(:,t+1), d);
            for j = 1:Nstate
                alpha(j,t+1) = alpha(:,t)'*A(:,j)*pyq(j,t+1);
            end
            alphasum = sum(alpha(:,t+1));
            alpha(:,t+1) = alpha(:,t+1)/alphasum;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % backward, beta
        beta(:,T) = ones(Nstate,1);
        betasum = sum(beta(:,T));
        beta(:,T) = beta(:,T) / betasum;
        for t = T-1:-1:1
            for i = 1:Nstate
                betatmp = 0;
                for j = 1:Nstate
                    betatmp = betatmp + beta(j,t+1)*pyq(j,t+1)*A(i,j);
                end
                beta(i,t) = betatmp;
            end
            betasum = sum(beta(:,t));
            beta(:,t) = beta(:,t)/betasum;
        end

        px = alpha(:,1)'*beta(:,1);
        totalLogLik = totalLogLik + log(px);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % posterior gamma
        gamma(:,T) = alpha(:,T);
        for t = T-1:-1:1
            for i = 1:Nstate
                gamma(i,t) = alpha(i,t)*beta(i,t)/px;
            end
            gamma(:,t) = gamma(:,t)/sum(gamma(:,t));
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % pairwise posterior xce
        for t = 1:T-1
            for i = 1:Nstate
                for j = 1:Nstate
                    xce(i,j,t) = alpha(i,t) * pyq(j,t+1) * A(i,j) * beta(j,t+1) / px;
                end
            end
            xce(:,:,t) = xce(:,:,t) / sum(sum(xce(:,:,t)));
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % accumulate
        PI_acc = PI_acc + gamma(:,1);

        A_num = A_num + sum(xce, 3);
        A_den = A_den + sum(gamma(:, 1:T-1), 2);

        gst = sum(gamma, 2);
        gamaSumT = gamaSumT + gst;
        gamaSumY = gamaSumY + y * gamma';

        for i = 1:Nstate
            for t = 1:T
                covAcc(:,:,i) = covAcc(:,:,i) + gamma(i,t) * (y(:,t)-uold(:,i))*(y(:,t)-uold(:,i))';
            end
        end
    end % end trace loop

    likeliHood = [likeliHood, totalLogLik];

    % check convergence
    if time >= 2
        if abs( likeliHood(time) - likeliHood(time-1) ) < criterion
            isConverged = true;
            U_op = u; Sigma_op = sigma; A_op = A; PI_op = PI;
        end
    end
    if time >= iter
        tooManyIteration = true;
        U_op = u; Sigma_op = sigma; A_op = A; PI_op = PI;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % M Step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % update PI
    PI = (PI_acc / sum(PI_acc))';

    % update transition matrix
    for i = 1:Nstate
        for j = 1:Nstate
            A(i,j) = A_num(i,j) / A_den(i);
        end
        A(i,:) = A(i,:) / sum(A(i,:));
    end

    % update constrained means
    a11 = zeros(size(Mg,2), size(Mg,2)); a12 = zeros(size(Mg,2), size(Mb,2)); a13 = zeros(size(Mg,2), size(Mr,2));
    a21 = zeros(size(Mb,2), size(Mg,2)); a22 = zeros(size(Mb,2), size(Mb,2)); a23 = zeros(size(Mb,2), size(Mr,2));
    a31 = zeros(size(Mr,2), size(Mg,2)); a32 = zeros(size(Mr,2), size(Mb,2)); a33 = zeros(size(Mr,2), size(Mr,2));
    for i = 1:Nstate
        a11 = a11 + Mg(:,:,i)' * sigma(:,:,i) * Mg(:,:,i) * gamaSumT(i);
        a12 = a12 + Mg(:,:,i)' * sigma(:,:,i) * Mb(:,:,i) * gamaSumT(i);
        a13 = a13 + Mg(:,:,i)' * sigma(:,:,i) * Mr(:,:,i) * gamaSumT(i);
        a21 = a21 + Mb(:,:,i)' * sigma(:,:,i) * Mg(:,:,i) * gamaSumT(i);
        a22 = a22 + Mb(:,:,i)' * sigma(:,:,i) * Mb(:,:,i) * gamaSumT(i);
        a23 = a23 + Mb(:,:,i)' * sigma(:,:,i) * Mr(:,:,i) * gamaSumT(i);
        a31 = a31 + Mr(:,:,i)' * sigma(:,:,i) * Mg(:,:,i) * gamaSumT(i);
        a32 = a32 + Mr(:,:,i)' * sigma(:,:,i) * Mb(:,:,i) * gamaSumT(i);
        a33 = a33 + Mr(:,:,i)' * sigma(:,:,i) * Mr(:,:,i) * gamaSumT(i);
    end
    c1 = zeros(size(Mg,2), 1);
    c2 = zeros(size(Mb,2), 1);
    c3 = zeros(size(Mr,2), 1);
    for i = 1:Nstate
        c1 = c1 + Mg(:,:,i)' * sigma(:,:,i) * gamaSumY(:,i);
        c2 = c2 + Mb(:,:,i)' * sigma(:,:,i) * gamaSumY(:,i);
        c3 = c3 + Mr(:,:,i)' * sigma(:,:,i) * gamaSumY(:,i);
    end
    coeff = [a11 a12 a13; a21 a22 a23; a31 a32 a33];
    b = [c1; c2; c3];
    results = coeff \ b;
    ug = results(1:length(c1));
    ub = results(length(c1)+1 : length(c1)+length(c2));
    ur = results(length(c1)+length(c2)+1 : length(c1)+length(c2)+length(c3));
    u = build_states_from_stateindex(stateindex, ug, ub, ur);

    % update covariances
    for i = 1:Nstate
        sigmatmp = covAcc(:,:,i) / gamaSumT(i);
        if sqrt(sigmatmp(1,1)) < 0.05 || sqrt(sigmatmp(2,2)) < 0.05 || sqrt(sigmatmp(3,3)) < 0.05 || det(sigmatmp) < 1.0e-9
            sigmatmp = sigma0(:,:,i);
        end
        sigma(:,:,i) = sigmatmp;
    end

end

end


