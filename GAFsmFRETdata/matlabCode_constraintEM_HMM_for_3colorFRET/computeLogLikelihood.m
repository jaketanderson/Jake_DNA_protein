function [totalLogLik, perTraceLogLik, totalT] = computeLogLikelihood(u, sigma, PI, A, ydataAll)
% Forward pass only. Returns sum of log p(y_n | theta) across traces, the
% per-trace log-likelihoods, and the total number of timepoints.

nTraces = length(ydataAll);
d = size(ydataAll{1}, 1);
Nstate = size(u,2);

totalLogLik = 0;
perTraceLogLik = zeros(1, nTraces);
totalT = 0;

for n = 1:nTraces
    y = ydataAll{n};
    T = size(y, 2);
    totalT = totalT + T;

    alpha = zeros(Nstate, T);
    pyq   = zeros(Nstate, T);
    logScale = 0;

    pyq(:,1) = gaussmmC(u, sigma, y(:,1), d);
    for i = 1:Nstate
        alpha(i,1) = PI(i) * pyq(i,1);
    end
    s = sum(alpha(:,1));
    alpha(:,1) = alpha(:,1) / s;
    logScale = logScale + log(s);

    for t = 1:T-1
        pyq(:,t+1) = gaussmmC(u, sigma, y(:,t+1), d);
        for j = 1:Nstate
            alpha(j,t+1) = alpha(:,t)'*A(:,j)*pyq(j,t+1);
        end
        s = sum(alpha(:,t+1));
        alpha(:,t+1) = alpha(:,t+1) / s;
        logScale = logScale + log(s);
    end

    perTraceLogLik(n) = logScale;
    totalLogLik = totalLogLik + logScale;
end

end
