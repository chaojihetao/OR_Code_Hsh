$ontext
         Author:18251115 黄世鸿
         Time: 2022/05/07
         Name: CG 列生成算法
$offtext
****模式表格，五类材料 i ;若干种切割模式 pp
set i/1,2,3,4,5,6/;
set pp/1*1000/;

****建立游标，方便记录和定位
set p(pp);
p('1')=yes;
p('2')=yes;
p('3')=yes;
p('4')=yes;
p('5')=yes;
set pi(pp);
pi('5')=yes;

parameters
num(i)/1 233,2 122,3 310,4 157,5 120/
wide(i)/1 18,2 21,3 91,4 136,5 51/
a(i,pp)/
1.1      1
2.2      1
3.3      1
4.4      1
5.5      1
/
w/274/
;

positive variable x(pp);
variable z_m;
integer variable y(i);
variable z_s;

**************** 建立主问题模型（集合覆盖类）
equations
obj_fuc
num_const(i)
;

obj_fuc..
         z_m =e= sum(p,x(p));
num_const(i)..
         sum(p,x(p)*a(i,p)) =g= num(i);
model master_model/obj_fuc,num_const/;

**************** 建立子问题模型（背包类）
equations
sub_fuc
wide_const
;

sub_fuc..
         z_s =e= 1-sum(i,num_const.m(i)*y(i));
wide_const..
         sum(i,wide(i)*y(i)) =l= w;
model sub_model/sub_fuc,wide_const/;

**************** 算法实现
set iter/1*20/;
parameter break_num/-1/;
loop(iter$(break_num < 0),
         solve master_model using LP minimizing z_m;
         solve sub_model using MIP minimizing z_s;
         break_num = z_s.l;
         pi(pp)=pi(pp-1);
         a(i,pi) = y.l(i);
         p(pi)=yes;
)
display z_m.l,x.l,z_s.l,a;

