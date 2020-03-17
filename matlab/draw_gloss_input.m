save_path = '/media/SENSETIME\qiudi/Data/relighting_download/results/analysisNet';
id = 20000;

D = im2double(imread([save_path, sprintf('/%d_depth.png', id)]));
I = im2double(imread([save_path, sprintf('/%d_rgb.png', id)]));
d = im2double(imread([save_path, sprintf('/%d_direction.png', id)]));

D = imresize(D, [128, 128]); D = repmat(D, [1,1,3]);
d = imresize(d, [128, 128]); d = repmat(d, [1,1,3]);

input = zeros(128*3, 256, 3);
input(1:256, 1:256, :) = I;
%%
input(257:128*3, 1:128, :) = D;
input(257:128*3, 129:256, :) = d;
imshow(input);
imwrite(input,[save_path sprintf('/%d_input.png', id)]);