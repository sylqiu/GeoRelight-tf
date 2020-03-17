function prepare_li_pointcloud(id)
path = '/home/likewise-open/SENSETIME/qiudi/Documents/nori_render/nori-costom/scenes/shape_17/';
[x, y] = meshgrid(1:256, 1:256);
flen = 128 / tan(pi/6);
x = (x - 128) / flen;
y = (y - 128) / flen;
% d = im2double(imread([path sprintf('%ddepth_2.png', id)]));
% d = d(:,:,1);
load([path sprintf('%ddepth.mat', id)]);
d = squeeze(d);
pc = cat(3, x.*d, y.*d, d);
normal = im2double(imread([path sprintf('%dnormal_2.png', id)]));
normal = (normal - 0.5) * 2;
save([path sprintf('%d_pc.mat', id)], 'pc');
save([path sprintf('%d_normal.mat', id)], 'normal');


end