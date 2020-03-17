function save_Li_result(i, save_path, BatchIndex)
i = int2str(i);
initAlbe = imread([i '_albedoPred_0.png']);
refineAlbe = imread([i '_albedoPred_2.png']);
initNormal = imread([i '_normalPred_0.png']);
refineNormal = imread([i '_normalPred_2.png']);
initRough = imread([i '_roughPred_0.png']);
refineRough = imread([i '_roughPred_2.png']);

ciA = crop_image(initAlbe, BatchIndex);
crA = crop_image(refineAlbe, BatchIndex);
ciN = crop_image(initNormal, BatchIndex);
crN = crop_image(refineNormal, BatchIndex);
ciR = crop_image(initRough, BatchIndex);
crR = crop_image(refineRough, BatchIndex);

imwrite(ciA, [save_path int2str(BatchIndex) '_init_albe.png']);
imwrite(crA, [save_path int2str(BatchIndex) '_refine_albe.png']);

imwrite(ciN, [save_path int2str(BatchIndex) '_init_normal.png']);
imwrite(crN, [save_path int2str(BatchIndex) '_refine_normal.png']);

imwrite(ciR, [save_path int2str(BatchIndex) '_init_mat.png']);
imwrite(crR, [save_path int2str(BatchIndex) '_refine_mat.png']);
end