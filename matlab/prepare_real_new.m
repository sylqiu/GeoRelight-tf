path = '/home/likewise-open/SENSETIME/qiudi/Documents/relighting_trial/tf_relight/losses/relight_img/real/tof_rgbd_relight_mask/';
out = '/home/likewise-open/SENSETIME/qiudi/Documents/relighting_trial/tf_relight/losses/relight_img/real/real_data/';
name = 'fruit';
fid = fopen([path name '/' name '_result/depth_more.bin']);
d2 = fread(fid,[141, 201], 'double');
% imshow(d, []); 
conf2 = im2double(imread([path name '/' name '_result/alpha.png']));
sz = size(d2);
xo = 10; yo = 30; s = 150;
% xo = 1; yo = 20; s = 130;

d2 = cat(1, d2, zeros(100, sz(2)));
conf2 = cat(1, conf2, zeros(100, sz(2)));

conf = imresize(conf2(xo:xo+s, yo:yo+s), [256,256], 'nearest');
conf(conf<1) = 0;
depth = imresize(d2(xo:xo+s, yo:yo+s), [256,256]) / 512 / 1.5;
imshow(depth, []);
save([out name '/depth.mat'], 'depth');