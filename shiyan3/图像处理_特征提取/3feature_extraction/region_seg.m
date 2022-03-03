mubiao1=imread('object1.png');
mubiao2=imread('object2.png');
figure(1),
subplot(2,2,1),imshow(mubiao1);
subplot(2,2,2),imshow(mubiao2);
eul1=bweuler(~mubiao1,8);S1=bwarea(mubiao1);
img1=bwperim(mubiao1,8);%求二值图中的边缘点
[m,n]=size(img1);
P=0;%周长初始化
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
img2=bwperim(mubiao2,8);%求二值图中的边缘点
[m,n]=size(img2);
P=0;%周长初始化
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
fprintf('目标1的欧拉数为 %8.1f\n',eul1),
fprintf('面积为 %8.2f\n',S1)
fprintf('周长为 %8.2f\n',L1)
fprintf('圆形度为 %8.5f\n',C1)
fprintf('形状复杂度为 %8.5f\n',e1)
fprintf('目标2的欧拉数为 %8.1f\n',eul2),
fprintf('面积为 %8.2f\n',S2)
fprintf('周长为 %8.2f\n',L2)
fprintf('圆形度为 %8.5f\n',C2)
fprintf('形状复杂度为 %8.5f\n',e2)