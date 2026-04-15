function [Mg, Mb, Mr] = build_selectMatrix(stateindex)

stateNum = size(stateindex,2);
d = size(stateindex,1);

% green color 
Mg = zeros(d, max(stateindex(1,:)), stateNum);
for i = 1:stateNum
    index = stateindex(1,i);
    Mg(1,index,i) = 1;
end

% blue color
Mb = zeros(d, max(stateindex(2,:)), stateNum);
for i = 1:stateNum
    index = stateindex(2,i);
    Mb(2,index,i) = 1;
end

% red color
Mr = zeros(d, max(stateindex(3,:)), stateNum);
for i = 1:stateNum
    index = stateindex(3,i);
    Mr(3,index,i) = 1;
end

end