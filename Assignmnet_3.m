%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
rng(5);
%setting up parameters

beta = 0.96;
gamma = 3.5;
r = 0.04;    
rho = 0.92;    
sigma = 0.05;

%setting up the probabilities
pu=0.05;
pe=0.75;

%Setting values for rows and col:

row_dim = 5000;
col_dim = 5;

%Doing Tauchen function for income and getting income

[y, P] = Tauchen(col_dim-1, 0, rho, sigma, 2);

income = exp(y);

%Determining the transition matrix

P_tran = zeros(col_dim);  % setting initial 5*5 of zeros

% Determining the first row for unemployment and transition to lowest employment probabilities 
P_tran(1,1) = 1 - pe;   % stay unemployed
P_tran(1,2) = pe;  % get to the lowest employment level

%Completing the transition matrix while adding the Tauchen probabilities in
%shifts between the income levels

% From employment states
for i = 2:col_dim
    P_tran(i,1) = pu; % each row has pu
    P_tran(i,2:col_dim) = (1 - pu) * P(i-1,:);
end


%setting up a_min and a_max
%a_min = 0;
a_min = -income(1)/r;
a_max = 10;
a_grid = linspace(a_min, a_max, row_dim);

%expanding the income to capture 0 for unemployment
income_unemp = [0; income];   % prepend 0 income for unemployment

% Predetermining w matrix
w = zeros(row_dim, col_dim);

%Calculating cash on hand
for j = 1:col_dim
    w(:, j) = a_grid + income_unemp(j);
end

%setting up the guess

c_1 = w;

%setting up parameters for while loop

diff = 1;
tol = 1e-9;       % convergence tolerance
max_it = 500;
it = 0;

c_new = c_1;

%looping

while diff > tol && it < max_it
    it = it + 1;
    c_old = c_new;
    
    % Expected marginal utility for each cash on hands
    EU = zeros(row_dim, col_dim);
    for j = 1:col_dim
        for j_next = 1:col_dim
            EU(:, j) = EU(:, j) + P_tran(j, j_next) * u_prime(c_old(:, j_next), gamma);
        end
    end
    
    % Calculating consumption useng the Euler equation
    c_calc = inv_u_prime(beta * (1+r) * EU, gamma);
     
    % Updating cash on hand
    w_next = a_grid' + c_calc;

    %Interpolating
    for j = 1:col_dim
         w_next(:, j) = real(w_next(:, j));  % Ensure w_next is real
         c_calc(:, j) = real(c_calc(:, j));   % Ensure c_calc is real
         c_new(:, j) = interp1(w_next(:, j), c_calc(:, j), w(:, j), 'pchip');
    end
  
    % Convergence check
    diff = max(abs(c_new(:) - c_old(:)));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PART B
figure;
hold on;

colors = lines(col_dim);  % color for each income state

for j = 1:col_dim
    plot(w(:, j), c_new(:, j), 'Color', colors(j,:));
end

xlabel('Cash-on-hand');
ylabel('Consumption');
title('Consumption Policy Function for Cash on Hand State');

grid on;
hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%PART C

%Simulating
T = 25; 
inc = zeros(T,1);
cons = zeros(T,1);

% Setting initial situation
a_t = 0;
emp_state = 1;

for t = 1:T
    % Record income
    inc(t) = income_unemp(emp_state);
    
    % Getting consumption based on the previous findings
    [a_period, a_dif] = min(abs(a_grid - a_t));
    cons(t) = c_new(a_dif, emp_state);
    
    % Updating the asset
    a_t = a_t + income_unemp(emp_state) - cons(t);
    
    % Ensure a_t stays within grid
    %a_t = max(min(a_t, a_max), a_min);
    
    % Transition between incomes
    emp_state = find(rand <= cumsum(P_tran(emp_state,:)), 1);
end

max_lag = 4;

% Compute cross-correlation
[cc, lags] = xcorr(cons, inc, max_lag, 'coeff');

% Plot correlogram
%figure;
%corrplot(cc, 'type', 'stem', lags);
%xlabel('Lag');
%ylabel('Correlation');
%title('Correlogram of Consumption and Income');
%grid on;


data = [cons, inc];  % each column is a variable

%names for labeling
varNames = {'Cons', 'Inc'};

% the correlation plot
figure;
corrplot(data, 'varNames', varNames);

title('Correlation');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Part E

%For the cash-on-hand model, it is linear because cash-on-hand consist of 
% assets and income and under CRRA utility it makes the consumer to be able
% smooth the consumtion despite the changes in income form the employment.
% In other words, capital helps to prevent drop in expenditures.

%
% For the VFI, it shows only relevance to the income (while exclusding the assets) 
% which is exposed to different high frequency employment probabilities 
% that can drastically lead to change in available income. This results 
% that for the lower income segments the value function is curved 
% exposing the higher risks associated with unemployment. 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%PART F

%Unemployment is more risky for the lower income groups exposing higher
%probabilities to become unemployed and get low assets. In order to escape
%the situation when there is no zero income in the next period, the
%consumer starts reducing consumption drastically that is resulting in the
%curvature.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Utility function:

function u_func = u(c, gamma)
    if gamma == 1
        u_func = log(c);
    else
        u_func = (c.^(1 - gamma)) ./ (1 - gamma);
    end
end

%%%%%%%%%%%%%%%%
%Getting a function for marginal utility
function up = u_prime(c, gamma)
    if gamma == 1
        up = 1 ./ c;
    else
        up = c.^(-gamma);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Inverse function for the marginal utility

function inv = inv_u_prime(up, gamma)
    if gamma == 1
        inv = 1 ./ up;
    else
        inv = up.^(-1/gamma);
    end
end