img1 = imread('object1.png');%原图
figure;
subplot(2,4,1);imshow(img1);title('原图')
Moment_Seven(img1);
I1=imrotate(img1,45,'bilinear');%旋转变化
subplot(2,4,2);imshow(I1);title('旋转');
Moment_Seven(I1);
I2=imresize(img1,2,'bilinear');%尺度变化
subplot(2,4,3);imshow(I2);title('尺度变化');
Moment_Seven(I2);
I3=flipdim(img1,2);%原图像的水平镜像
subplot(2,4,4);imshow(I3);title('镜像变化');
Moment_Seven(I3);

img2 = imread('object2.png');%原图


subplot(2,4,5);imshow(img2);
Moment_Seven(img2);title('原图')
I5=imrotate(img2,45,'bilinear');%旋转变化
subplot(2,4,6);imshow(I5);title('旋转');
Moment_Seven(I5);
I6=imresize(img2,2,'bilinear');%尺度变化
subplot(2,4,7);imshow(I6);title('尺度变化');
Moment_Seven(I6);
I7=flipdim(img2,2);%原图像的水平镜像
subplot(2,4,8);imshow(I7);title('镜像变化');
Moment_Seven(I7);

    
 