function [psnr_old, loss_img] = plot_psnr(loss, m, M, idx, idx2, c1, c2, bar_flag)
fid = fopen('/home/likewise-open/SENSETIME/qiudi/Documents/relighting_trial/utils/Dirs.txt');
list = textscan(fid, '%f %f %f\n');
list = cell2mat(list);
as = 40;
%%
% figure(1);
% plot(list(:, 1), list(:, 2), 'o');
%%
% fid = fopen('/home/likewise-open/SENSETIME/qiudi/Documents/Deep-Relighting/out/test/refine_targets/deg90_5/out/alllosses.txt');
% loss = textscan(fid, '%f\n');
% loss = cell2mat(loss);
% loss = reshape(loss, 1053, [])';

%%
% loss = mean(loss, 1)';
%%
% close all;
% load('resultsloss.mat');
% loss = loss'*255^2;
% loss = loss(39,:)';
psnr = log10(255)*20 - log10(loss)*10;
% mean(psnr(:))
psnr_old = psnr;
% psnr = mean(psnr, 2);
% psnr = psnr_old(:, 52);
%
% m = 18;
% M = 30;
psnr(psnr > M) = M;
psnr(psnr < m) = m;
psnr = 1 - (psnr - m) / (M-m);
psnr = round(psnr*63 + 1);
%
cmap = colormap('jet');
psnr_color = cmap(psnr, :);

%
[xx, yy] = meshgrid(1:540, 1:540);
xx = (xx - 270) / 270;
yy = (yy - 270) / 270;
id = (xx.^2 + yy.^2) > 0.9;
id = repmat(id, 1, 1, 3);
%
ss = 8;
loss_img = zeros(540, 540, 3);

for i = 1:1052
   px = round((list(i, 1) + 1) / 2 * 499) + 1 + 16;
   py = 500 - (round((list(i, 2) + 1) / 2 * 499 + 1)) + 16;
   loss_img(py:py+ss-1, px:px+ss-1, :) = repmat(reshape(psnr_color(i, :), [1,1,3]), [ss, ss, 1]);
   if idx + 1 == i
       x = px+as+ss/2; y = py+as+ss/2;
   elseif idx2 + 1 == i
       x2 = px+as+ss/2; y2 = py+as+ss/2;
   end
    
end
%
loss_img(id) = 1;
figure(1);

imshow(loss_img); hold on;
quiver(x,y,-as,-as, 0, 'Linewidth', 8, 'Color', c1, 'MaxHeadSize', 8);
quiver(x2,y2,-as,-as, 0, 'Linewidth', 8, 'Color', c2, 'MaxHeadSize', 8);
% 
if bar_flag == 1
colormap(cmap(end:-1:1, :));
colorbar('southoutside');
caxis([m, M]);

end
end