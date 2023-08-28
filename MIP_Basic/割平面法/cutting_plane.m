clc;
clear all;
c = [0;-1];
b = [6;0];
As = [3,2;-3,2];
Ag = [];
while 1
    [A,As,b,c,X,z]=BigMMethod(As,Ag,b,c);
    isinteger = 1;
    for i = 1:length(X)
        if abs(X(1,i) - round(X(1,i))) > 1.0e-7
            isinteger = 0;
            q = i;
            break
        end
    end
    if isinteger == 1
       X
       A
       return
    end
    Af = A(q,:) - floor(A(q,:));
    bf = b(q,:) - floor(b(q,:));
    As = A;
    Ag = Af;
    b = [b;bf];
end