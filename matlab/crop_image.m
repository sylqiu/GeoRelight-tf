function cI = crop_image(I, BatchIndex)
row = 1 + floor(BatchIndex / 8) * 256;
col = 1 + mod(BatchIndex, 8)*256;
row = row + 2*(floor(BatchIndex / 8) + 1);
col = col + 2*(mod(BatchIndex, 8) + 1);
if length(size(I)) == 3
    cI = I(row:row+255, col:col+255, :);
%     cI = cI(2:end-2, 2:end-2,:);
else
    cI = I(row:row+255+4, col:col+255+4);
end


end