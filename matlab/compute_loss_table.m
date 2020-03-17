clear;
names = {'', 'noisydepth', 'env'};
losses = {'normal', 'mat_l2', 'albe_l2'};
pre = '/home/likewise-open/SENSETIME/qiudi/Documents/relighting_trial/tf_relight/losses';
loss_table = zeros(3,3);
for i = 1:3
    load([pre '/nohl_' names{i} '_losses/resultsloss_' losses{1} '.mat']) 
    load([pre '/nohl_' names{i} '_losses/resultsloss_' losses{2} '.mat']) 
    load([pre '/nohl_' names{i} '_losses/resultsloss_' losses{3} '.mat']) 
    loss_table(i,1) = mean(loss_albe_l2);
    loss_table(i,3) = mean(loss_mat_l2);
    loss_table(i,2) = mean(loss_normal);
end