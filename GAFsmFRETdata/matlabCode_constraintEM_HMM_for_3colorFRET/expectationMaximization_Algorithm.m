
function[U_op, Sigma_op, A_op, likeliHood, PI_op] = expectationMaximization_Algorithm(stateindex, u, sigma, PI, A, y)
%%%%%%%%%%
% function for computing posterior probabilities of HMM using EM algorithm
% impose gaussian probabilities on emission probabilities,
% parametirized by u and covariance matrix sigma. 

% Math is based on the book: 
% Bishop- Pattern Recognition and Machine Learning - Springer (2006) 
% Chapter 13

% observed data: y(t).  states: q(k).

% iterate the algorithm for convergence

likeliHood = [];%zeros(1,iter);

sigma0 = sigma;

[d, T] = size(y); % T timepoints, d: dimensionality of data. 
Nstate = size(u,2); % state number

pyq   = zeros(Nstate, T);     % emission probability: p(y|q)
alpha = zeros(Nstate, T);     % forward
beta  = zeros(Nstate, T);
gamma = zeros(Nstate, T);     % posterior
xce   = zeros(Nstate, Nstate,T-1);  % segment posterior, transition

[Mg, Mb, Mr] = build_selectMatrix(stateindex); % selectionMatrix for the constraint EM algorithm

criterion = 1e-3;
isConverged = false;
iter = 100;
tooManyIteration = false;
time = 0;
while isConverged == false && tooManyIteration == false 
    
    time = time + 1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % E step
    
    %PI = PI;  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % forward, alpha
    % Initialize alpha,     
    % first compute emission probability at time-point 1
    pyq(:,1) = gaussmmC(u, sigma, y(:,1), d);
    
    for i = 1:Nstate
       alpha(i,1) = PI(i) * pyq(i,1); % Eq 13.37
    end
    alphasum = sum(alpha(:,1));
    alpha(:,1) = alpha(:,1)/alphasum;  % normalize

    for t = 1:T-1
        pyq(:,t+1) = gaussmmC(u, sigma, y(:,t+1), d);
        for j = 1:Nstate
            alpha(j,t+1) = alpha(:,t)'*A(:,j)*pyq(j,t+1); % Eq 13.36
        end
        alphasum = sum(alpha(:,t+1));
        alpha(:,t+1) = alpha(:,t+1)/alphasum;  %normalize  
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % backward, beta, Eq 13.38
    beta(:,T) = ones(Nstate,1);
    betasum = sum(beta(:,T));
    beta(:,T) = beta(:,T) / betasum; % normalize
    for t = T-1:-1:1
        for i = 1:Nstate
            betatmp = 0;
            for j = 1:Nstate
                betatmp = betatmp + beta(j,t+1)*pyq(j,t+1)*A(i,j);
            end
            beta(i,t) = betatmp;
        end
        betasum = sum(beta(:,t));
        beta(:,t) = beta(:,t)/betasum; % normalize
    end 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update the likelihood
    % likelihood, Eq 13.41-13.42
    % px = alpha(:,T)'*beta(:,T) ;
    px = alpha(:,1)'*beta(:,1) ;
    likeliHood = [likeliHood, px];%(time) = px;
    %%%%%%%%%%%%%%%%%%%%%%%
    % check whether to stop
    %
    if time >= 2
        if abs( likeliHood(time) - likeliHood(time-1) ) < criterion
            isConverged = true;
            U_op = u; Sigma_op = sigma; A_op = A; PI_op = PI;
        end
    end
    %
    if time >= iter
        tooManyIteration = true;
        U_op = u; Sigma_op = sigma; A_op = A; PI_op = PI;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % backward, gamma, posterior probabilites, Eq 13.33
    gamma(:,T) = alpha(:,T);
    for t = T-1:-1:1
        for i = 1:Nstate
            gamma(i,t) = alpha(i,t)*beta(i,t)/px;
        end
        gamma(:,t) = gamma(:,t)/sum(gamma(:,t));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %compute cooccurence recursions, Eq 13.43
    for t = 1:T-1
        for i = 1:Nstate
            for j = 1:Nstate
                xce(i,j,t) = alpha(i,t) * pyq(j,t+1) * A(i,j) * beta(j,t+1) / px;
            end
        end
        xce(:,:,t) = xce(:,:,t) / sum(sum(xce(:,:,t))); % normalize
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % M Step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update PI, initial emission prob. Eq 13.18
    for i = 1:Nstate
       PI(i) = gamma(i,1) / sum(gamma(:,1)); % actually, gamma is already normalized
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %update transition matrix, Eq 13.19 
    for i = 1:Nstate
        xcesum = sum(sum(xce(i,:,:)));
        for j = 1:Nstate
            A(i,j) = sum(xce(i,j,:))/xcesum;  
        end
        A(i,:) = A(i,:)/sum(A(i,:)); % normalize
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update emission probs
    uold = u;
    % first update ub and ur
    gamaSumT = sum(gamma,2); % sum gama on T axis
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
    gamaSumY = y * gamma';
    c1 = zeros(size(Mg,2), 1);
    c2 = zeros(size(Mb,2), 1);
    c3 = zeros(size(Mr,2), 1);
    for i = 1:Nstate
        c1 = c1 + Mg(:,:,i)' * sigma(:,:,i) * gamaSumY(:,i);
        c2 = c2 + Mb(:,:,i)' * sigma(:,:,i) * gamaSumY(:,i);
        c3 = c3 + Mr(:,:,i)' * sigma(:,:,i) * gamaSumY(:,i);
    end
    % solve the system of linear equations
    coeff = [a11 a12 a13; a21 a22 a23; a31 a32 a33];
    b = [c1; c2; c3];
    results = coeff \ b; 
    ug = results(1:length(c1));
    ub = results(length(c1)+1 : length(c1)+length(c2));
    ur = results(length(c1)+length(c2)+1 : length(c1)+length(c2)+length(c3));
    % rebuilt the system U according to stateIndex and the values in ub and ur
    u = build_states_from_stateindex(stateindex, ug, ub, ur);
    
    %update sigmas, Eq 13.21
    for i = 1:Nstate
        del = zeros(d,d);
        for j = 1:T
            del = del + gamma(i,j) * (y(:,j)-uold(:,i))*(y(:,j)-uold(:,i))';
        end
        sigmatmp = del/gamaSumT(i); % each sigma is d*d matrix
%
        % make sure each sigma has inverse
        if sqrt(sigmatmp(1,1)) < 0.05 || sqrt(sigmatmp(2,2)) < 0.05 || sqrt(sigmatmp(3,3)) < 0.05 || det(sigmatmp) < 1.0e-9 
            FRETa = 0.15;
            sigmatmp = [FRETa^2, 0.0, 0.0
                        0.0, FRETa^2, 0.0
                        0.0, 0.0, FRETa^2];
            sigmatmp = sigma0(:,:,i);
        end  
%
        sigma(:,:,i) = sigmatmp; % each sigma is d*d matrix   
    end
    %pause
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %go back to start until done iterating
end

end


