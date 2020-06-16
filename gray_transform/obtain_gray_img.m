%% convert RGB to Gray
%% augment DUTS dataset spliting RGB image to R, G and B channel and gt is the same
img_dir = './img/';
% gt_dir = '/home/jing-zhang/jing_file/RGB_sal_dataset/train/DUTS/gt/';
save_img_dir = './gray/';
% save_gt_dir = '/home/jing-zhang/jing_file/RGB_sal_dataset/train/DUTS_Gray/gt1/';
img_list = dir([img_dir '*' 'jpg']);

for i = 1:length(img_list)
    i
    img_cur = imread([img_dir img_list(i).name]);
%     gt_cur = imread([gt_dir img_list(i).name(1:end-4) '.png']);
    if(size(img_cur,3) == 3)
        img_r = img_cur(:,:,1);
        img_g = img_cur(:,:,2);
        img_b = img_cur(:,:,3);
    elseif (size(img_cur,3) == 1)
        img_r = img_cur;
        img_g = img_cur;
        img_b = img_cur;
    end
    
    img_r_linear = gamma_expansion(img_r);
    img_g_linear = gamma_expansion(img_g);
    img_b_linear = gamma_expansion(img_b);
    
    gray_cur = 0.2126*img_r_linear + 0.7152*img_g_linear + 0.0722*img_b_linear;
    gray_cur_uint8 = uint8(gray_cur*255);
    
    imwrite(gray_cur_uint8, [save_img_dir img_list(i).name(1:end-4) '.png']);
   
%     imwrite(gt_cur,[save_gt_dir img_list(i).name(1:end-4) '_gray.png']);
%     
%     source = [img_dir img_list(i).name];
%     target = [save_img_dir img_list(i).name];
%     copyfile(source,target);
%     
%     source = [gt_dir img_list(i).name(1:end-4) '.png'];
%     target = [save_gt_dir img_list(i).name(1:end-4) '.png'];
%     copyfile(source,target);
end