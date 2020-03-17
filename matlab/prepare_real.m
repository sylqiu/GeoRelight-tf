function prepare_real(name)
path = '/home/likewise-open/SENSETIME/qiudi/Documents/relighting_trial/tf_relight/losses/relight_img/real/tof_rgbd_relight_mask/';
out = '/home/likewise-open/SENSETIME/qiudi/Documents/relighting_trial/tf_relight/losses/relight_img/real/real_data/';
conf2 = im2double(imread([path name '/' name '_result/alpha.png']));
rgb2 = im2double(imread([path name '/' name '_result/img.png']));
d2 = im2double(imread([path name '/' name '_result/depth_gaussian.png']));
sz = size(d2);
% xo = 10; yo = 30; s = 150;
xo = 1; yo = 10; s = 140;
rgb2 = cat(1, rgb2, zeros(100, sz(2), 3));
d2 = cat(1, d2, zeros(100, sz(2)));
conf2 = cat(1, conf2, zeros(100, sz(2)));

conf = imresize(conf2(xo:xo+s, yo:yo+s), [256,256], 'nearest');
conf(conf<1) = 0;
rgb = imresize(rgb2(xo:xo+s, yo:yo+s, :), [256,256]);
depth = imresize(d2(xo:xo+s, yo:yo+s), [256,256]);
imshow(rgb);
% [x, y] = meshgrid(1:256, 256:-1:1);
% flen = 128 / tan(pi/9);
% x = (x - 128) / flen;
% y = (y - 128) / flen;
% pc = cat(2, x(:).* d(:), y(:) .* d(:), -d(:));
% Pc = pointCloud(pc);
% pcwrite(Pc, 'test.ply');
mkdir([out name]);
imwrite(rgb .* repmat(reshape(conf, [256,256,1]), [1,1,3]), [out name '/rgb_flash.png']);
imwrite(conf, [out name '/conf.png']);
save([out name '/depth.mat'], 'depth');
end