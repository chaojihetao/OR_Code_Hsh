function [x_opt,z_opt,v,w,z_dual]=func_dual_simplex(A,b,c)
[m,n]=size(A);
J=prod(1:n);    %%%%%%%%求n的阶乘，方便后续进行遍历
all_v=perms(1:n);%%%%%%%%将所有的v列举出
%%%%%%%% 1.判断对偶解是否可行
for j=1:J          
    v=all_v(j,:);
    B=A(:,v(1:m));
    N=A(:,v(m+1:n));
    cB=c(v(1:m),:);
    cN=c(v(m+1:n),:);
    r=cN'-(cB'*inv(B))*N;   %%%%%%%% 对偶可行等价all(r>=0)
    if all(r>=0)
        break
    end
end 
%%%%%%%% 2.判断原问题解是否可行
 x0=inv(B)*b;
v;
z=cB'*x0;
w=cB'*inv(B);
w=w';
z_dual=b'*w;
if all(x0>=0)       %%%%%%%% 依据强弱对偶定理，原问题与对偶问题都可行时，所得解为最优解
   sprintf("x is optimal")
   x0=inv(B)*b;
    v;
    z=cB'*x0;
 w=cB'*inv(B);
    w=w';
    z_dual=b'*w;
   x_opt=x0;
   z_opt=cB'*x0;
   return
end
%%%%%%%% 3.最优解判断后，进行换基，步进得最优解
while 1 
    if min(x0)>=0
        sprintf("x is optimal")
        x_opt=x0;
        z_opt=cB'*x0;
     w=cB'*inv(B);
       w=w';
    z_dual=b'*w;
        return
    end
    [~,q]=min(x0);      %%%%%%%% 找xq<0
    out=q;
    for i=m+1:n         %%%%%%%% 遍历所有非基变量
        in=i;
        temp=v(1,out);
        v(1,out)=v(1,in);
        v(1,in)=temp;
        B=A(:,v(1:m));
        N=A(:,v(m+1:n));
        cB=c(v(1:m),:);
        cN=c(v(m+1:n),:);
        r=cN'-(cB'*inv(B))*N;
        if all(r>=0)  %%%%%%%% 注意这里的巧妙之处：只需要再次令对偶解可行，就能找到最优解
            break
        end
    end
    x0=inv(B)*b;
    v;
    z=cB'*x0;
    w=cB'*inv(B);
    w=w';
    z_dual=b'*w;
end
end
