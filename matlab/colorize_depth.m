function color_A = colorize_depth(path)
% A = im2double(imread(path));
A = load([path 'depth.mat']);
conf = im2double(imread([path 'conf.png']));
if length(size(conf)) == 3
   conf = conf(:,:,1); 
end
A = A.depth;
% mask = A==0;
% mask = mask .* (1 - conf);
mask = 1 - conf;
A = A(:);
B = A(~mask);
A = (A - min(B)) ./ (max(B) - min(B));
A = A .* (1-mask(:));
cmap = colormap('jet');
color_A = cmap(round(A*63)+1,:);
color_A = reshape(color_A, [256, 256, 3]) .* repmat(1 - mask, [1,1,3]);
imshow(color_A);
% imwrite(color_A, [path '_color.png']);
imwrite(color_A, [path 'depth_color.png']);
end