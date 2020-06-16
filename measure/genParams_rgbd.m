function p = genParams_rgbd()
%% Generate environment and user-specific parameters.
p.salObjSets = {'NJU2K','STERE','DES','NLPR','LFSD','SIP'};
p.GTsuffix = {'.png','.png','.png','.png','.png','.png'};
p.Imgsuffix = {'.png','.png','.png','.png','.png','.png'};


p.salObjSets = p.salObjSets(:);
setNum = length(p.salObjSets);
%% set p.algMapDir as your own saliency map directory
p.algMapDir = '/home/jing-zhang/jing_file/CVPR2020/RGBD/final_model/results/';
%% p.GTDir is ground truth saliency map directory
p.GTDir = '/home/jing-zhang/jing_file/CVPR2020/RGBD/test/';  %% gt ile path

%%
end