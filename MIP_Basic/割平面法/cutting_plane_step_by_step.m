
clc;
As=[1,-3;1,1];
Ag=[];
b=[1;4];
c=[-2;0];
x=linspace(0,5,30);y=(b(1,1)-As(1,1)*x)/As(1,2);
plot(x,y,'g');hold on;
x=linspace(0,5,30);y=(b(2,1)-As(2,1)*x)/As(2,2);
plot(x,y,'g');hold on;
[m,n]=size(b);
[A,As,b,c,X,z]=BigMMethod(As,Ag,b,c);

for i=0:5
    x=i*[1,1,1,1,1,1];
    y=[0,1,2,3,4,5];
    plot(x,y,'.');
    hold on;
end

for i=1:m
    if b(i)-fix(b(i))>0.00000001
        q=i;
    end
end

while q~=0
    Af=A(q,:)-floor(A(q,:))
    bf=b(q)-floor(b(q))
    As=A
    Ag=Af;
    b=[b;bf]
    
    An=As(1,:);bn=b(1,:);
    for i=1:4
        An(1,i)=An(1,i)+As(2,i)*(-2);
        
    end
    An=An-Af
    bn=1
    x=linspace(0,5,30);y=(bn(1,1)-An(1,1)*x)/An(1,2);
    plot(x,y,'r');hold on;
    
    [A,As,b,c,X,z]=BigMMethod(As,Ag,b,c);
    [m,n]=size(b);
    q=0;
    for i=1:m
        if b(i)-fix(b(i))>0.00000001
            q=i;
        end
    end
end
X
z








