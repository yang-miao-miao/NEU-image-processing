I=imread('ab.jpg');
I=rgb2gray(I);
figure(1); 
imshow(I);  
title('原图');
figure(2);
imhist(I);
title('直方图');
figure(3);
%人工设置阈值
t=140;
%可以尝试不同的阈值选取方法
%bw=graythresh(I);  
%disp(strcat('otsu阈值分割的阈值:',num2str(bw*255)));%在command window里显示出 :迭代的阈值:阈值 

I1=imbinarize(I,t/255); %图像二值化 
imshow(I1);
 
 
%保存图像（最好可以交互设定文件名和目录）
%filepath1=('D:\MATLAB\bin\Project\Moment_invariants\mubiao1.png')
%imwrite(quyu1,filepath1);