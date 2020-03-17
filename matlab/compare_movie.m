function compare_movie(path_me, path_xu, flag)
writerObj = VideoWriter([path_me 'animate.avi']);
writerObj.FrameRate = 20;
open(writerObj);
if  strcmp(flag, 'dir')
    light_path_me = '/home/likewise-open/SENSETIME/qiudi/Documents/Deep-Relighting/out/CircleRender/deg90_5/0/lightImg_circle_90_circle/';
    light_path_xu = '/home/likewise-open/SENSETIME/qiudi/Documents/Deep-Relighting/out/CircleRender/deg90_5/17/lightImg_circle_90_circle/';
elseif strcmp(flag, 'grace')
    light_path_xu = '/home/likewise-open/SENSETIME/qiudi/Documents/Deep-Relighting/out/envRender/deg90_5/0/lightImg_env_90_grace_probe.pfm/';

elseif strcmp(flag,'stpeters')
    light_path_xu = '/home/likewise-open/SENSETIME/qiudi/Documents/Deep-Relighting/out/envRender/deg90_5/17/lightImg_env_90_stpeters_probe.pfm/';

end

for i = 0:199
    
        img_me = imresize(0.9*imread([path_me sprintf('%d_rgb_pred.png', i)]), [256,256]);
%         img_me = imresize(1.5*imread([path_me sprintf('%04d.png', i)]), [256,256]);
        img_xu = imresize(imread([path_xu sprintf('%04d.png', i)]), [256,256]);
%         img = imread([path sprintf('%d_rgb_pred.png', i)]);
%         img = imrotate(img, 270);
%         img = imresize(img(1:220, 1+20:1+20+219, :), [256,256]);
        
   
    img_me = cat(1, img_me, zeros(25, 256, 3));
    img_me = cat(2, img_me, zeros(256+25, 25, 3));
    img_xu = cat(1, img_xu, zeros(25, 256, 3));
    img_xu = cat(2, img_xu, zeros(256+25, 25, 3));
    if strcmp(flag, 'dir')
        light_img_xu = imresize(imread([light_path_xu sprintf('%04d.png', i)]), [50, 50]);
        light_img_me = imresize(imread([light_path_me sprintf('%04d.png', i)]), [50, 50]);
        img_xu(256+1-25:256+25, 256+1-25:256+25, :) = light_img_xu;
        img_me(256+1-25:256+25, 256+1-25:256+25, :) = light_img_me;
    else
        light_img = imresize(imread([light_path_xu sprintf('%04d.png', i)]), [50, 50]);
        img_me(256+1-25:256+25, 256+1-25:256+25, :) = light_img;
    end
    img = cat(2, img_me, img_xu);
    
    frame = im2frame(img);
    writeVideo(writerObj, frame);
end
close(writerObj);
end