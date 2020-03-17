function crop_loss_img(img)
path = '/home/likewise-open/SENSETIME/qiudi/Documents/relighting_trial/tf_relight/losses/loss_img/';
A = imread([path img '.png']);
x = 512;
B = A(40:40+x, 109:109+x,:);
% imshow(B);
imwrite(B, [path img '_crop.png']);
end