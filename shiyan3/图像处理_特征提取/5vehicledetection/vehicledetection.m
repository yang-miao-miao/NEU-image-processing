clear;clc;
I=imread('1.jpg');       %读取当前路径下的图片
%%%%%%%%%%%  HSV颜色分割图像  %%%%%%%%%%%%%%%%%%%%%%%%%
hsvImg=rgb2hsv(I);%转换到HSV空间
h1=hsvImg(:,:,1);%H分量
s1=hsvImg(:,:,2);%S分量
v1=hsvImg(:,:,3);%V分量
%提取红色分量   
hsvReg1=((h1<=0.05&h1>=0)|(h1>=0.78&h1<=1.0))&s1>=0.05&s1<=1.0&v1>=0.0005&v1<=1.0;%此处为HSV空间红色范围
figure,subplot(1,2,1),
imshow(hsvReg1);title('原图hsv检测图像');
%寻找连通区域 
imLabel = bwlabel(hsvReg1);
stats = regionprops(hsvReg1,'Area');    %求各连通域的大小
area = cat(1,stats.Area);
index = find(area == max(area));%寻找最大面积连通区域
img = ismember(imLabel,index);
subplot(1,2,2),imshow(img);,title('最大连通区域') 
img_reg = regionprops(img,  'boundingbox'); 
rects = cat(1,  img_reg.BoundingBox);  
%显示连通区域 
figure(2),  
imshow(I);
for i = 1:size(rects, 1)  
    if( rects(i,3)>20||rects(i,4)>20)
    rectangle('position', rects(i, :), 'EdgeColor', 'b');  
    end
end  


