function Moment_Seven(J)                  %J为要求解的图像
A=double(J);                               %将图像数据转换为double类型
[m,n]=size(A);                              %求矩阵A的大小
[x,y]=meshgrid(1:n,1:m);                     %生成网格采样点的数据，x,y的行数等于m，列数等于n
x=x(:);                                     %矩阵赋值    
y=y(:);                                           
A=A(:);                                     
m00=sum(A);                               %求矩阵A中每列的和，得到m00是行向量
if m00==0                                  %如果m00=0，则赋值m00=eps，即m00=0
    m00=eps;
end
m10=sum(x.*A);                             %以下为7阶矩求解过程，参见7阶矩的公式
m01=sum(y.*A);
xmean=m10/m00;
ymean=m01/m00;
cm00=m00;
cm02=(sum((y-ymean).^2.*A))/(m00^2);
cm03=(sum((y-ymean).^3.*A))/(m00^2.5);
cm11=(sum((x-xmean).*(y-ymean).*A))/(m00^2);
cm12=(sum((x-xmean).*(y-ymean).^2.*A))/(m00^2.5);
cm20=(sum((x-xmean).^2.*A))/(m00^2);
cm21=(sum((x-xmean).^2.*(y-ymean).*A))/(m00^2.5);
cm30=(sum((x-xmean).^3.*A))/(m00^2.5);
Mon(1)=cm20+cm02;                        %1阶矩Mon(1)
Mon(2)=(cm20-cm02)^2+4*cm11^2;           %2阶矩Mon(2)
Mon(3)=(cm30-3*cm12)^2+(3*cm21-cm03)^2;  %3阶矩Mon(3)
Mon(4)=(cm30+cm12)^2+(cm21+cm03)^2;     %4阶矩Mon(4)
Mon(5)=(cm30-3*cm12)*(cm30+cm12)*((cm30+cm12)^2-3*(cm21+cm03)^2)+(3*(cm30+cm12)^2-(cm21+cm03)^2);                                        %5阶矩Mon(5)
Mon(6)=(cm20-cm02)*((cm30+cm12)^2-(cm21+cm03)^2)+4*cm11*(cm30+cm12)*(cm21+cm03); %6阶矩Mon(6)
Mon(7)=(3*cm21-cm03)*(cm30+cm12)*((cm30+cm12)^2-3*(cm21+cm03)^2)+(3*cm12-cm30)*(cm21+cm03)*(3*(cm30+cm12)^2-(cm21+cm03)^2);             %7阶矩Mon(7)
qijieju=abs(log(Mon))                      %采用log函数缩小不变矩的动态范围值

