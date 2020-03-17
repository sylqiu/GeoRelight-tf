function write_normal()
out = '/home/likewise-open/SENSETIME/qiudi/Documents/relighting_trial/tf_relight/losses/testing_2020-03-10-17-22_relight_xu_fulldir_wsrc/results/';
load([out '/58_normal.mat']);
normal2 = (normal + 1)/2;
% mask = (normal > 0) * 1.0;
normal2(normal2 == 0.5) = 0;
imwrite(normal2, [out '/58_normal.png']);
% imwrite(normal, [out '/1_normal.png']);
end