function u = build_states_from_stateindex(stateindex, ug, ub, ur)

u = stateindex * 0;
for i = 1:size(stateindex,2)
    u(1,i) = ug(stateindex(1,i));
    u(2,i) = ub(stateindex(2,i));
    u(3,i) = ur(stateindex(3,i));
end

end