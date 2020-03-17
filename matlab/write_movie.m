function write_movie(path, flag)
writerObj = VideoWriter([path 'animate.avi']);
writerObj.FrameRate = 20;
open(writerObj);
if strcmp(flag, 'me')
    light_path = '/home/likewise-open/SENSETIME/qiudi/Documents/Deep-Relighting/out/CircleRender/deg90_5/0/lightImg_circle_90_circle/';
elseif strcmp(flag, 'xu')
    light_path = '/home/likewise-open/SENSETIME/qiudi/Documents/Deep-Relighting/out/CircleRender/deg90_5/17/lightImg_circle_90_circle/';
elseif strcmp(flag, 'grace')
    light_path = '/home/likewise-open/SENSETIME/qiudi/Documents/Deep-Relighting/out/envRender/deg90_5/0/lightImg_env_90_grace_probe.pfm/';
elseif strcmp(flag,'stpeters')
    light_path = '/home/likewise-open/SENSETIME/qiudi/Documents/Deep-Relighting/out/envRender/deg90_5/17/lightImg_env_90_stpeters_probe.pfm/';
end
for i = 0:199
    
    if strcmp(flag, 'me')
        img = imresize(imread([path sprintf('%d_rgb_pred.png', i)]), [256,256]);
    else
%         img = imresize(imread([path sprintf('%04d.png', i)]), [256,256]);
%         img = imread([path sprintf('%d_rgb_pred.png', i)]);
        img = imread([path sprintf('%04d.png', i)]);
        img = imrotate(img, 270);
        img = imresize(img(1:220, 1+20:1+20+219, :), [256,256]);
        
    end
    img = cat(1, img, zeros(25, 256, 3));
    img = cat(2, img, zeros(256+25, 25, 3));
    light_img = imresize(imread([light_path sprintf('%04d.png', i)]), [50, 50]);
    light_img = imrotate(light_img, 270);
    img(256+1-25:256+25, 256+1-25:256+25, :) = light_img;
    frame = im2frame(img);
    writeVideo(writerObj, frame);
end
close(writerObj);
end