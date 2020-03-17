function brighten_img(id)
path = '/home/likewise-open/SENSETIME/qiudi/Documents/nori_render/nori-costom/scenes/pa3/';
out = '/home/likewise-open/SENSETIME/qiudi/Documents/relighting_trial/tf_relight/losses/relight_img/';

% imwrite(3*im2double(imread([path sprintf('%d_sips_env91.png', id)])), [out sprintf('%d_sips_env91.png', id)]);
% imwrite(1.1*im2double(imread([out '0091.png'])), [out sprintf('%d_my_env91.png', id)]);
% imwrite(imresize(1.5*im2double(imread([out '0091_xu.png'])), [256,256]), [out sprintf('%d_xu_env91.png', id)]);
% imwrite(1.4*im2double(imread([path sprintf('%d_sips_dir638.png', id)])), [out sprintf('%d_sips_dir638.png', id)]);
% imwrite(1.4*im2double(imread([path sprintf('%d_sips_dir450.png', id)])), [out sprintf('%d_sips_dir450.png', id)]);
% imwrite(2*im2double(imread([path sprintf('%d_sips_env1.png', id)])), [out sprintf('%d_sips_env1.png', id)]);
imwrite(1.1*im2double(imread([out '0001.png'])), [out sprintf('%d_my_env1.png', id)]);
% imwrite(1.4*im2double(imread([out '0001_xu.png'])), [out sprintf('%d_xu_env1.png', id)]);

end