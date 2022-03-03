clear; 
clc; 
src = imread('ab2.png'); %读取文件数据
figure(1);  
subplot(3,3,1),imshow(src); 
title('原始图像 ');%显示原图

%用ostu方法获取二值化阈值，进行二值化并进行显示
level=graythresh(src);
bw=im2bw(src,level);
subplot(3,3,2),imshow(bw),title('二值图像')
 
%运用开操作消去噪点
se = strel('disk',2);
openbw=imopen(bw,se);%对白色点而言
subplot(3,3,3),imshow(openbw),title('开运算后的效果图')
 
%获取连通区域，并进行显示
% L = bwlabel(openbw,8);
[L,num] = bwlabel(openbw,8);
RGB = label2rgb(L);
subplot(3,3,4),imshow(RGB),title('用rgb颜色标记不同区域')

stats = regionprops(openbw, 'basic');
centroids = cat(1, stats.Centroid);
subplot(3,3,5),imshow(openbw);
for i=1:size(stats)
      rectangle('Position',[stats(i).BoundingBox],'LineWidth',2,'LineStyle','-','EdgeColor','r');
     stats(i).BoundingBox
    
    
end
     stats(2).BoundingBox(1)
     abs(stats(2).BoundingBox(3)-stats(2).BoundingBox(1))
     object1 = imcrop(openbw,[stats(2).BoundingBox(1),stats(2).BoundingBox(2),abs(stats(2).BoundingBox(3)),abs(stats(2).BoundingBox(4))]);
     subplot(3,3,6),imshow(object1),title('object1');
     %filepath1=('F:\matlab\bin\project\实验题目3&要求-特征提取\图像处理_特征提取\4Moment_invariants\object1.png')
     %filepath3=('F:\matlab\bin\project\实验题目3&要求-特征提取\图像处理_特征提取\3feature_extraction\object1.png')
     %imwrite(object1,filepath3);
     %imwrite(object1,filepath1);
     
     object2=imcrop(openbw,[stats(3).BoundingBox(1),stats(3).BoundingBox(2),abs(stats(3).BoundingBox(3)),abs(stats(3).BoundingBox(4))]);
     subplot(3,3,7),imshow(object2),title('object1');
     %filepath2=('F:\matlab\bin\project\实验题目3&要求-特征提取\图像处理_特征提取\4Moment_invariants\object2.png')
     %imwrite(object2,filepath2);
      %filepath4=('F:\matlab\bin\project\实验题目3&要求-特征提取\图像处理_特征提取\3feature_extraction\object2.png')
     %imwrite(object2,filepath4);


