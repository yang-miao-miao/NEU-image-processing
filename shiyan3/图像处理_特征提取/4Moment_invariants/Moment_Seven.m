function Moment_Seven(J)                  %JΪҪ����ͼ��
A=double(J);                               %��ͼ������ת��Ϊdouble����
[m,n]=size(A);                              %�����A�Ĵ�С
[x,y]=meshgrid(1:n,1:m);                     %�����������������ݣ�x,y����������m����������n
x=x(:);                                     %����ֵ    
y=y(:);                                           
A=A(:);                                     
m00=sum(A);                               %�����A��ÿ�еĺͣ��õ�m00��������
if m00==0                                  %���m00=0����ֵm00=eps����m00=0
    m00=eps;
end
m10=sum(x.*A);                             %����Ϊ7�׾������̣��μ�7�׾صĹ�ʽ
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
Mon(1)=cm20+cm02;                        %1�׾�Mon(1)
Mon(2)=(cm20-cm02)^2+4*cm11^2;           %2�׾�Mon(2)
Mon(3)=(cm30-3*cm12)^2+(3*cm21-cm03)^2;  %3�׾�Mon(3)
Mon(4)=(cm30+cm12)^2+(cm21+cm03)^2;     %4�׾�Mon(4)
Mon(5)=(cm30-3*cm12)*(cm30+cm12)*((cm30+cm12)^2-3*(cm21+cm03)^2)+(3*(cm30+cm12)^2-(cm21+cm03)^2);                                        %5�׾�Mon(5)
Mon(6)=(cm20-cm02)*((cm30+cm12)^2-(cm21+cm03)^2)+4*cm11*(cm30+cm12)*(cm21+cm03); %6�׾�Mon(6)
Mon(7)=(3*cm21-cm03)*(cm30+cm12)*((cm30+cm12)^2-3*(cm21+cm03)^2)+(3*cm12-cm30)*(cm21+cm03)*(3*(cm30+cm12)^2-(cm21+cm03)^2);             %7�׾�Mon(7)
qijieju=abs(log(Mon))                      %����log������С����صĶ�̬��Χֵ

