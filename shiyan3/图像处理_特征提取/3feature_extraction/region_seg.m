mubiao1=imread('object1.png');
mubiao2=imread('object2.png');
figure(1),
subplot(2,2,1),imshow(mubiao1);
subplot(2,2,2),imshow(mubiao2);
eul1=bweuler(~mubiao1,8);S1=bwarea(mubiao1);
img1=bwperim(mubiao1,8);%���ֵͼ�еı�Ե��
[m,n]=size(img1);
P=0;%�ܳ���ʼ��
for i=1:m
    for j=1:n
        if(img1(i,j)>0)
            P=P+1;
        end
    end
end
L1=P;
C1=4*pi*S1/power(L1,2);
e1=power(L1,2)/S1;
eul2=bweuler(~mubiao2,8);S2=bwarea(mubiao2);
img2=bwperim(mubiao2,8);%���ֵͼ�еı�Ե��
[m,n]=size(img2);
P=0;%�ܳ���ʼ��
for i=1:m
    for j=1:n
        if(img2(i,j)>0)
            P=P+1;
        end
    end
end
L2=P ;
C2=4*pi*S2/power(L2,2);
e2=power(L2,2)/S2;
fprintf('Ŀ��1��ŷ����Ϊ %8.1f\n',eul1),
fprintf('���Ϊ %8.2f\n',S1)
fprintf('�ܳ�Ϊ %8.2f\n',L1)
fprintf('Բ�ζ�Ϊ %8.5f\n',C1)
fprintf('��״���Ӷ�Ϊ %8.5f\n',e1)
fprintf('Ŀ��2��ŷ����Ϊ %8.1f\n',eul2),
fprintf('���Ϊ %8.2f\n',S2)
fprintf('�ܳ�Ϊ %8.2f\n',L2)
fprintf('Բ�ζ�Ϊ %8.5f\n',C2)
fprintf('��״���Ӷ�Ϊ %8.5f\n',e2)