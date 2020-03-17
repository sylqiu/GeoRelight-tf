function loss = compute_individual_loss(BatchIndex)
nohl_pre = './testing_2019-12-27-20-43_relight_taxo_nohl/results/imgs/';
env_pre = './testing_2019-12-28-10-28_relight_taxo_nohl_env/results/imgs/';
nd_pre = './testing_2019-12-28-10-28relight_taxo_nohl_noisydepth/results/imgs/';
li_pre = './Li_individual/';

gt_albe = im2double(imread([nohl_pre int2str(BatchIndex) '_albe.png']));
gt_normal = im2double(imread([nohl_pre int2str(BatchIndex) '_normal.png']));
conf = sum(abs(gt_normal), 3) == 0;
conf = conf * 1.0;
% gt_normal = convert_to_normal(gt_normal, conf);
gt_mat = im2double(imread([nohl_pre int2str(BatchIndex) '_mat.png']));

nohl_albe = im2double(imread([nohl_pre int2str(BatchIndex) '_albepred.png']));
nohl_normal = im2double(imread([nohl_pre int2str(BatchIndex) '_normalpred.png']));
nohl_mat = im2double(imread([nohl_pre int2str(BatchIndex) '_matpred.png']));
% nohl_normal = convert_to_normal(gt_normal, conf);

env_albe = im2double(imread([env_pre int2str(BatchIndex) '_albepred_env.png']));
env_normal = im2double(imread([env_pre int2str(BatchIndex) '_normalpred_env.png']));
env_mat = im2double(imread([env_pre int2str(BatchIndex) '_matpred_env.png']));

nd_albe = im2double(imread([nd_pre int2str(BatchIndex) '_albepred_noisydepth.png']));
nd_normal = im2double(imread([nd_pre int2str(BatchIndex) '_normalpred_noisydepth.png']));
nd_mat = im2double(imread([nd_pre int2str(BatchIndex) '_matpred_noisydepth.png']));

li_albe_init = im2double(imread([li_pre int2str(BatchIndex) '_init_albe.png']));
li_normal_init = im2double(imread([li_pre int2str(BatchIndex) '_init_normal.png']));
li_mat_init = mean(im2double(imread([li_pre int2str(BatchIndex) '_init_mat.png'])),3);
li_albe_refine = im2double(imread([li_pre int2str(BatchIndex) '_refine_albe.png']));
li_normal_refine = im2double(imread([li_pre int2str(BatchIndex) '_refine_normal.png']));
li_mat_refine = mean(im2double(imread([li_pre int2str(BatchIndex) '_refine_mat.png'])),3);

loss = [compute_conf_loss(gt_albe, nohl_albe, conf) / 3,...
    compute_conf_loss(gt_normal, nohl_normal, conf) / 3,...
    compute_conf_loss(gt_mat, nohl_mat, conf),...
    compute_conf_loss(gt_albe, env_albe, conf) / 3,...
    compute_conf_loss(gt_normal, env_normal, conf) / 3,...
    compute_conf_loss(gt_mat, env_mat, conf),...
    compute_conf_loss(gt_albe, nd_albe, conf) / 3,...
    compute_conf_loss(gt_normal, nd_normal, conf) / 3,...
    compute_conf_loss(gt_mat, nd_mat, conf),...
    compute_conf_loss(gt_albe, li_albe_init, conf) / 3,...
    compute_conf_loss(gt_normal, li_normal_init, conf) / 3,...
    compute_conf_loss(gt_mat, li_mat_init, conf),...
    compute_conf_loss(gt_albe, li_albe_refine, conf) / 3,...
    compute_conf_loss(gt_normal, li_normal_refine, conf) / 3,...
    compute_conf_loss(gt_mat, li_mat_refine, conf)];

end