% given the emissionProb and transitionProb, we can determine the state sequence 
% using the Viterbi Algorithm to find the most likely hidden state sequence
function [stateSequence, probMax, probSequence] = stateSequence_Viterbi_Algorithm(y, Umatrix, sigma2, transitionProbMatrix, PI)
stateNum = size(Umatrix,2);

timeSteps = size(y,2);
stateSequence = zeros(1,timeSteps); % state index for each time point
probSequence = zeros(1,timeSteps);
probMatrix = zeros(timeSteps,stateNum); % max prob at time i with abservation j
indexBacktrack = zeros(timeSteps,stateNum);

for i = 1:timeSteps
    for j = 1:stateNum
        % emission prob 
        % emission prob is the product of the multi-color data at the time step
        ydata = y(:,i);
        u = Umatrix(:,j);
        sigma = sigma2(:,:,j);
        eptmp = 1/(2*pi)/sqrt(det(sigma)) * exp(-1/2* (ydata-u)'*(inv(sigma))*(ydata-u));
        %eptmp = 1 * exp(-1/2* (ydata-u)'*(inv(sigma))*(ydata-u));

        %{
        if i == 1
            probMatrix(i,j) = log(eptmp * PI(j));
            indexBacktrack(i,j) = j;
        else
            probMaxTmp = -1e8;
            probMaxIndex = 1;
            for k = 1:stateNum 
                if transitionProbMatrix(k,j) < 1e-8 % consider the efficient transitions
                    probtmp = log(eptmp) + probMatrix(i-1,k) + (-Inf);
                else
                    probtmp = log(eptmp) + probMatrix(i-1,k) + log(transitionProbMatrix(k,j));
                end
                if probtmp > probMaxTmp
                    probMaxTmp = probtmp;
                    probMaxIndex = k;
                end
            end
            probMatrix(i,j) = probMaxTmp;
            indexBacktrack(i,j) = probMaxIndex;
        end
        %}
        
        if i == 1
            probMatrix(i,j) = eptmp * PI(j);
            indexBacktrack(i,j) = j;
        else
            probMaxTmp = -1e8;
            probMaxIndex = 1;
            for k = 1:stateNum 
                probtmp = eptmp * probMatrix(i-1,k) * transitionProbMatrix(k,j);
                if probtmp > probMaxTmp
                    probMaxTmp = probtmp;
                    probMaxIndex = k;
                end
            end
            probMatrix(i,j) = probMaxTmp;
            indexBacktrack(i,j) = probMaxIndex;
        end
    end
end
for i = timeSteps:-1:1 % back-tracking to find the most probable path
    if i == timeSteps
        probtmp = probMatrix(i,:);
        [probMax, index] = max(probtmp);
        stateSequence(i) = index;
        probSequence(i) = probMax / sum(probtmp);
    else
        stateSequence(i) = indexBacktrack(i+1,stateSequence(i+1));
        probtmp = probMatrix(i,:);
        probSequence(i) = probMatrix(i,stateSequence(i)) / sum(probtmp);
    end
end

end