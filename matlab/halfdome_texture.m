function halfdome_texture(id)
path = '/home/likewise-open/SENSETIME/qiudi/Documents/Deep-Relighting/out/envRender/deg90_5/17/lightImg_env_90_stpeters_probe.pfm';
A = im2double(imread([path sprintf('/%04d.png', id)]));
% imshow(A);
img = zeros(64,128,3);
for t = 1:32
    for p = 1:128
    theta = (t-0.5)/64 * pi;
    phi = (p-0.5)/64 * pi;
    r = sin(theta);
    xpos = floor(r*cos(phi) * 31.5 + 31.5)+1;
    ypos = floor(-r*sin(phi) * 31.5 + 31.5)+1;
    img(t, p,:) = A(ypos, xpos, :);
    end
end
imshow(imresize(img, 4));
imwrite(img, [path sprintf('/p/%04d_p.png', id)]);
end