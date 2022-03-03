clear;clc;
I=imread('1.jpg');       %��ȡ��ǰ·���µ�ͼƬ
%%%%%%%%%%%  HSV��ɫ�ָ�ͼ��  %%%%%%%%%%%%%%%%%%%%%%%%%
hsvImg=rgb2hsv(I);%ת����HSV�ռ�
h1=hsvImg(:,:,1);%H����
s1=hsvImg(:,:,2);%S����
v1=hsvImg(:,:,3);%V����
%��ȡ��ɫ����   
hsvReg1=((h1<=0.05&h1>=0)|(h1>=0.78&h1<=1.0))&s1>=0.05&s1<=1.0&v1>=0.0005&v1<=1.0;%�˴�ΪHSV�ռ��ɫ��Χ
figure,subplot(1,2,1),
imshow(hsvReg1);title('ԭͼhsv���ͼ��');
%Ѱ����ͨ���� 
imLabel = bwlabel(hsvReg1);
stats = regionprops(hsvReg1,'Area');    %�����ͨ��Ĵ�С
area = cat(1,stats.Area);
index = find(area == max(area));%Ѱ����������ͨ����
img = ismember(imLabel,index);
subplot(1,2,2),imshow(img);,title('�����ͨ����') 
img_reg = regionprops(img,  'boundingbox'); 
rects = cat(1,  img_reg.BoundingBox);  
%��ʾ��ͨ���� 
figure(2),  
imshow(I);
for i = 1:size(rects, 1)  
    if( rects(i,3)>20||rects(i,4)>20)
    rectangle('position', rects(i, :), 'EdgeColor', 'b');  
    end
end  


