I=imread('ab.jpg');
I=rgb2gray(I);
figure(1); 
imshow(I);  
title('ԭͼ');
figure(2);
imhist(I);
title('ֱ��ͼ');
figure(3);
%�˹�������ֵ
t=140;
%���Գ��Բ�ͬ����ֵѡȡ����
%bw=graythresh(I);  
%disp(strcat('otsu��ֵ�ָ����ֵ:',num2str(bw*255)));%��command window����ʾ�� :��������ֵ:��ֵ 

I1=imbinarize(I,t/255); %ͼ���ֵ�� 
imshow(I1);
 
 
%����ͼ����ÿ��Խ����趨�ļ�����Ŀ¼��
%filepath1=('D:\MATLAB\bin\Project\Moment_invariants\mubiao1.png')
%imwrite(quyu1,filepath1);