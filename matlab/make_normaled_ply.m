function make_normaled_ply(path)
load([path '_pc.mat']); load([path '_normal.mat']);

% new_pc = [];
% new_normal = [];
fid = fopen([path '_pc2.ply'], 'w');
% fid_uv = fopen([path '_uv.obj'], 'w');
fprintf(fid, 'ply\nformat ascii 1.0\nelement vertex 24628\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nend_header\n');
for i = 1:size(pc, 1)
    for j = 1:size(pc, 2)
        tx = (j-1) / 255.0;
        ty = 1 - (i-1)/255.0;
%         ux = (j - 128)/256;
%         uy = -1*(i - 128)/256;
        if sum(abs(pc(i, j, 3))) > 0 && sum(abs(normal(i,j,:))) > 0
%             new_pc = [new_pc; pc(i, j, :) .* reshape([1, -1, -1], [1,1,3])];
%             new_normal = [new_normal; normal(i,j,:)];  
            fprintf(fid, '%f %f %f %f %f %f\n',...
                pc(i, j, 1), pc(i,j,2)*-1, pc(i,j,3)*-1,...
                normal(i,j,1), normal(i,j,2), normal(i,j,3));
%             fprintf(fid_uv, 'v %f %f %f\n', pc(i, j, 1), pc(i,j,2)*-1, pc(i,j,3)*-1);
%             fprintf(fid_uv, 'vn %f %f %f\n',normal(i,j,1), normal(i,j,2), normal(i,j,3));
%             fprintf(fid_uv, 'vt %f %f\n', tx, ty);
            
%             pc(i, j, 3) * ux / pc(i, j, 1)
        end
    end
end

% P = pointCloud(new_pc, 'Normal', new_normal);
% pcwrite(P, [path '_pc.ply']);
end