function [z_opt,x_opt] = Branch_and_bround_01(A,b,c)

%     clc;
%     clear all;
%     A = [6,8,10,13];
%     c = [18;23;30;40];
%     b = [25];
%     Aeq = []; beq = [];
    [~,n] = size(A);
    xl = zeros(1,n);
    xu = ones(1,n);
    A_M = [1,0,0,0,0,0];
    now_level = 1;
    all_level = 1;
    z_limit = -inf;
while now_level <= all_level
    if A_M(now_level,1) == 0
        now_level = now_level + 1;
        continue
    end
    x_level = A_M(now_level,2);
    Aeq = [];beq = [];
    for xp=1:x_level
        new_row=zeros(1,n);
        new_row(1,xp)=1;
        Aeq=[Aeq;new_row];
        beq=[beq;A_M(now_level,xp+2)];
    end    
    options = optimoptions( 'linprog', 'Display', 'none' );
    [x,z,exit]=linprog(-c,A,b,Aeq,beq,xl,xu,options);
    z = -z;
    if exit == -2   % no solution
        A_M(now_level,1) = 0;
    elseif z < z_limit % not optimal
        A_M(now_level,1) = 0;
    elseif z >= z_limit && max(abs(x-round(x)))>=1.0e-7 % not feasible
        A_M(now_level,1) = 0;    % branch
        all_level = all_level + 2;
        x_level = A_M(now_level,2) + 1;
        branch_L=A_M(now_level,:) ;branch_R=A_M(now_level,:) ;
        branch_L(1,1)=1;branch_R(1,1)=1;
        branch_L(1,x_level+2)=1; 
        branch_R(1,x_level+2)=0;
        branch_L(1,2) = x_level;branch_R(1,2) = x_level;
        A_M = [A_M;branch_L;branch_R];
    elseif z >= z_limit && max(abs(x-round(x)))<1.0e-7  % feasible
        A_M(now_level,1) = 0;
        z_limit = z;
        x_opt = x;
    end
    now_level = now_level + 1 ;
end
z_opt = z_limit;
end
