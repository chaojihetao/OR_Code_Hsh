function [z_opt,x_opt] = Branch_and_bround(A,b,c,Aeq,beq)
clc;
%   Time:2022/04/22
%   Author:18251115 黄世鸿
%   Name:Branch_and_bround

% A = [9,7;7,20];
% c = [40;90];
% b = [56;70];
% Aeq = []; beq = [];
[m,n] = size(A);
xl = zeros(1,n);
xu = inf(1,n);
A_M = [1,0,0,0];
now_level = 1;
all_level = 1;
z_limit = -inf;
while now_level <= all_level
    x_level = A_M(now_level,2);
    A_add = [] ;b_add = [];
    for k = 1:x_level
       row = zeros(1,n);
       if A_M(now_level,k+2)>=0
           row(1,k)=1;
           A_add = [A_add;row];
           A = [A;A_add];
           b_add = [b_add;A_M(now_level,k+2)];
           b = [b;b_add];
       elseif A_M(now_level,k+2)<0
           row(1,k)=-1;
           A_add = [A_add;row];
           A = [A;A_add];
           b_add = [b_add;-A_M(now_level,k+2)];
           b = [b;b_add];
       end
    end
    options = optimoptions( 'linprog', 'Display', 'none' );
    [x,z,exit]=linprog(-c,A,b,Aeq,beq,xl,xu,options);
    z = -z;
    for i=1:m
       A_M(now_level,i+2)=x(i); 
    end
    if exit == -2   % no solution
        A_M(now_level,1) = 0;
    elseif z < z_limit % not optimal
        A_M(now_level,1) = 0;
    elseif z >= z_limit && max(abs(x-round(x)))>=1.0e-7 % not feasible
        A_M(now_level,1) = 0;    % branch
        all_level = all_level + 2;
        x_level = A_M(now_level,2) + 1;
        branch_L=A_M(now_level,:) ;branch_R=A_M(now_level,:) ;
        branch_L(1,x_level+2)=ceil(A_M(now_level,x_level+2))-1 ; 
        branch_R(1,x_level+2)=-ceil(A_M(now_level,x_level+2));
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

