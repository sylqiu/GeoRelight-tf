function draw_light_pos(id)
num = [id];
fid = fopen('/home/likewise-open/SENSETIME/qiudi/Documents/relighting_trial/utils/Dirs.txt');
list = textscan(fid, '%f %f %f\n');
list = cell2mat(list);
loss_img = zeros(540, 540);
ss = 15;
[xx, yy] = meshgrid(1:540, 1:540);
xx = (xx - 270) / 270;
yy = (yy - 270) / 270;
id = (xx.^2 + yy.^2) > 0.9;
for i = 1:1052
   if ismember(i, num);
   px = round((list(i, 1) + 1) / 2 * 499) + 1 + 16;
   py = 500 - (round((list(i, 2) + 1) / 2 * 499 + 1)) + 16;
   loss_img(py-ss:py+ss-1, px-ss:px+ss-1) = 1;
   end
end
loss_img(id) = 1;
% imshow(loss_img);
% imwrite(loss_img, ['/home/likewise-open/SENSETIME/qiudi/Documents/relighting_trial/tf_relight/losses/loss_img/' sprintf('%d_indir.png', id)]);
imwrite(loss_img, './test.png');
end