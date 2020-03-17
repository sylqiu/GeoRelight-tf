function add_textcoord(path)
[f, v] = gpp_read_obj([path '_mesh2.obj']);
[f, v, ~] = clean_mesh(f, v);
load([path '_pc.mat']); load([path '_normal.mat']);
xyz = reshape(pc, [256*256, 3]);
xyz = xyz .* repmat(reshape([1, -1, -1], [1, 3]), [256*256, 1]);
normal = reshape(normal, [256*256, 3]);

% fid = fopen([path '_uv.obj']);
% uv_list = textscan(fid, '%s %f %f %f\n%s %f %f\n');
% uv = cell2mat(uv_list([2,3,4]));
% text_uv = cell2mat(uv_list([6,7]));
[text_u, text_v] = meshgrid(0:255, 255:-1:0);
text_u = (text_u + 0.5)/255;
text_v = (text_v + 0.5)/255;
text_uv = cat(2, text_u(:), text_v(:));
textcoord = [];
new_normal = [];
for i = 1:size(v, 1)
%     [lia, locb] = ismember(v(i,:), uv, 'rows');
%     if sum(lia) >= 1
%         
%     end 
%     [lia, locb] = ismember(v(i,:), xyz, 'rows');
    diff = sum(abs(xyz - repmat(v(i,:), [size(xyz,1), 1])), 2);
    lia = diff < 0.00001;
     if sum(lia) >= 1
        new_normal = [new_normal; normal(lia,:)];
        textcoord = [textcoord; text_uv(lia,:)];
    end 
end
% write into new obj
fid2 = fopen([path '_uvc.obj'], 'w');
fprintf(fid2, 'v %f %f %f\n', v');
fprintf(fid2, 'vt %f %f\n', textcoord');
fprintf(fid2, 'vn %f %f %f\n', new_normal');
fprintf(fid2, 'f %d/%d/%d %d/%d/%d %d/%d/%d\n', [f(:,1), f(:,1), f(:,1), f(:,2), f(:,2), f(:,2)...
    , f(:,3), f(:,3), f(:,3)]');

end