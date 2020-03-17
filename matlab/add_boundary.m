function add_boundary(img, c)
[path, name, ~] = fileparts(img); 
A = im2double(imread(img));
if size(size(A)) == 2
A = repmat(A, [1,1,3]);
end
if size(A,1) ~= 256
   A = imresize(A, [256, 256]); 
end
mask = zeros(256, 256, 1);
w = 6;
for i = 1:256
    for j = 1:256
        if i <= w || i >= 256 - w
             mask(i,j) = 1; 
        end
        if j <= w || j >= 256 - w
               mask(i,j) = 1; 
        end
        
    end
end
% imshow(mask);
mask = repmat(mask, [1,1,3]); 
A = A .* (1 - mask) + mask .* repmat(reshape(c, [1, 1, 3]), [256, 256, 1]);
% imshow(A);
imwrite(A, [path '/' name '_frame.png']);
end