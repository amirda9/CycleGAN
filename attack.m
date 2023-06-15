mpc = loadcase('case118');

Vm = 0.95 + rand(118,1)*0.1;
Va = -180 + rand(118, 1)*360;


mpc.bus(:, 8) = Vm;
mpc.bus(:, 9) = Va;

% disp(mpc.bus(5,8))

% Generate a random state for the system using AC power flow analysis
results = runopf(mpc);

default_demand_p = mpc.bus(:,3);
default_demand_q = mpc.bus(:,4);
grid_size_full = size(default_demand_p);
grid_size = grid_size_full(1,1);

bus_has_load_p = default_demand_p>0;
bus_has_load_q = default_demand_q>0;

default_cost_c1 = mpc.gencost(:,6);
default_cost_c2 = mpc.gencost(:,5);

gen_size = size(default_cost_c1);

measures = [];
states = [];

mpc = loadcase('case118');

beta1 = 1;
beta2 = 1;

% Modify the voltage measurements and run power flow simulation
Vm = 0.95 + rand(118,1)*0.1;
Va = -180 + rand(118, 1)*360;
mpc.bus(:, 8) = Vm;
mpc.bus(:, 9) = Va;

rdm_per_p = ones(grid_size,1)*(-0.2)+rand(grid_size,1)*0.4;
rdm_per_q = ones(grid_size,1)*(-0.2)+rand(grid_size,1)*0.4;
new_p = default_demand_p.*(1+rdm_per_p).*bus_has_load_p;
new_q = default_demand_q.*(1+rdm_per_q).*bus_has_load_q;
mpc.bus(:,3) = new_p;
mpc.bus(:,4) = new_q;
% 
%     new_c1 = mpc.gencost(:,6)*0.1+rand(gen_size).*mpc.gencost(:,6);
%     new_c2 = mpc.gencost(:,5)*0.1+rand(gen_size).*mpc.gencost(:,5);
%     
%     mpc.gencost(:,6) = new_c1;
%     mpc.gencost(:,5) = new_c2;
    

results = runopf(mpc);
    



meas = [results.bus(:,3).',results.bus(:,4).',results.branch(:,14).',results.branch(:,15).'];
state = [Vm.',Va.'];
disp(state)

measures = [measures;meas];
states = [states;state];


attack_region_buses = [4,5,11,12,13];
attack_region_lines = [3,10,11,12,14,16];

meas_attacked = meas;

% // attack
% // attack on the measurements
for i=1:length(attack_region_buses)
    meas_attacked(attack_region_buses(i)) = (0.8*meas_attacked(attack_region_buses(i))) + (0.4*rand(1)*meas_attacked(attack_region_buses(i)));
    meas_attacked(attack_region_buses(i)+118) = (0.8*meas_attacked(attack_region_buses(i)+118)) + (0.4*rand(1)*meas_attacked(attack_region_buses(i)+118));
end

for i=1:length(attack_region_lines)
    meas_attacked(attack_region_lines(i)+ 236) = (0.8*meas_attacked(attack_region_lines(i)+ 236)) + (0.4*rand(1)*meas_attacked(attack_region_lines(i)+ 236));
    meas_attacked(attack_region_lines(i)+ 422) = (0.8*meas_attacked(attack_region_lines(i)+ 422)) + (0.4*rand(1)*meas_attacked(attack_region_lines(i)+ 422)); 
end

% state estimation using the attacked measurements

mpc.bus(:,3) = meas_attacked(1:118);
mpc.bus(:,4) = meas_attacked(119:236);


results_attacked = runopf(mpc);

states_attacked = [results_attacked.bus(:,8).',results_attacked.bus(:,9).'];
meas_attacked = [results_attacked.bus(:,3).',results_attacked.bus(:,4).',results_attacked.branch(:,14).',results_attacked.branch(:,15).'];

beta1cost = 0;
for j = 1:length(attack_region_buses)
    beta1cost = beta1cost + abs(states_attacked(attack_region_buses(j)) - state(attack_region_buses(j)));
    beta1cost = beta1cost + abs(states_attacked(attack_region_buses(j)+118) - state(attack_region_buses(j)+118));
end
beta2cost = 0;
for j = 1:length(attack_region_lines)
    beta2cost = beta2cost + abs(meas_attacked(attack_region_lines(j)) - meas(attack_region_lines(j)));
    beta2cost = beta2cost + abs(meas_attacked(attack_region_lines(j)+186) - meas(attack_region_lines(j)+186));
end

cost_init = beta1 * beta1cost - beta2 * beta2cost;
disp(cost_init)

%     // gradient descent
h = 0.0001;


loss = []
cost = cost_init;
for i=1:10
%// zero order gradient

meas_temp = meas_attacked;
    for k = 1:length(attack_region_buses)
        meas_temp(attack_region_buses(k)) = meas_temp(attack_region_buses(k)) + h;
        meas_temp(attack_region_buses(k)+118) = meas_temp(attack_region_buses(k)+118) + h;
    end
mpc.bus(:,3) = meas_temp(1:118);
mpc.bus(:,4) = meas_temp(119:236);
%     for k = 1:length(attack_region_lines)
%         mpc.branch(attack_region_lines(k),14) = mpc.branch(attack_region_lines(k),14) + h;
%         mpc.branch(attack_region_lines(k),15) = mpc.branch(attack_region_lines(k),15) + h;
%     end
    results = runopf(mpc);
    states_h =  [results.bus(:,8).',results.bus(:,9).'];
    measures_h = [results.bus(:,3).',results.bus(:,4).',results.branch(:,14).',results.branch(:,15).'];

    beta1cost_h = 0;
    for j = 1:length(attack_region_buses)
        beta1cost_h = beta1cost_h + abs(states_h(attack_region_buses(j)) - states(attack_region_buses(j)));
        beta1cost_h = beta1cost_h + abs(states_h(attack_region_buses(j)+118) - states(attack_region_buses(j)+118));
    end
    beta2cost_h = 0;
     for j = 1:length(attack_region_lines)
         beta2cost_h = beta2cost_h + abs(measures_h(attack_region_lines(j)+236) - meas(attack_region_lines(j)+236));
         beta2cost_h = beta2cost_h + abs(measures_h(attack_region_lines(j)+422) - meas(attack_region_lines(j)+422));
     end
    
    cost_h = beta1 * beta1cost_h - beta2 * beta2cost_h;
    
%     // first order gradient
    grad = (cost_h - cost)/h;

%     // update the measurements
    for k = 1:length(attack_region_buses)
        meas_temp(attack_region_buses(k)) = meas_temp(attack_region_buses(k)) - grad*0.001;
        meas_temp(attack_region_buses(k)+118) = meas_temp(attack_region_buses(k)+118) - grad*0.001;
    end
mpc.bus(:,3) = meas_temp(1:118);
mpc.bus(:,4) = meas_temp(119:236);
%     for k = 1:length(attack_region_lines)
%         mpc.branch(attack_region_lines(k),14) = mpc.branch(attack_region_lines(k),14) - grad;
%         mpc.branch(attack_region_lines(k),15) = mpc.branch(attack_region_lines(k),15) - grad;
%     end
    results = runopf(mpc);
    states_attacked =  [results.bus(:,8).',results.bus(:,9).'];
    measures_attacked = [results.bus(:,3).',results.bus(:,4).',results.branch(:,14).',results.branch(:,15).'];
    

    beta1cost_after = 0;
    for j = 1:length(attack_region_buses)
        beta1cost_after = beta1cost_after + abs(states_attacked(attack_region_buses(j)) - states(attack_region_buses(j)));
        beta1cost_after = beta1cost_after + abs(states_attacked(attack_region_buses(j)+118) - states(attack_region_buses(j)+118));
    end
    beta2cost_after = 0;
    for j = 1:length(attack_region_lines)
        beta2cost_after = beta2cost_after + abs(measures_attacked(attack_region_lines(j)+236) - measures(attack_region_lines(j)+236));
        beta2cost_after = beta2cost_after + abs(measures_attacked(attack_region_lines(j)+422) - measures(attack_region_lines(j)+422));
    end
    cost = beta1 * beta1cost_after - beta2 * beta2cost_after;
    loss = [loss, cost];

end

for i=1:length(attack_region_buses)
    disp(meas_attacked(attack_region_buses(i)))
    disp(meas(attack_region_buses(i)))
    disp(meas_attacked(attack_region_buses(i)+118))
    disp(meas(attack_region_buses(i)+118))
end

for i=1:length(attack_region_lines)
    disp(meas_attacked(attack_region_lines(i)+236))
    disp(meas(attack_region_lines(i)+236))
    disp(meas_attacked(attack_region_lines(i)+422))
    disp(meas(attack_region_lines(i)+422))
end
plot(loss)

# append the results to a csv file
csvwrite('meas_attacked.csv', meas_attacked)
csvwrite('states_attacked.csv', states_attacked)
csvwrite('meas.csv', meas)




