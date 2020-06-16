
Thresholds = 1:-1/255:0;
p = genParams_rgbd();
smeasure_total = zeros(length(p.salObjSets),1);
maxF_total = zeros(length(p.salObjSets),1);
meanF_total = zeros(length(p.salObjSets),1);
maxE_total = zeros(length(p.salObjSets),1);
meanE_total = zeros(length(p.salObjSets),1);
mae_total = zeros(length(p.salObjSets),1);
f_total = {};
e_total = {};


for curSet = 1:6
	curSetName = p.salObjSets{curSet};
    salPath = [p.algMapDir curSetName '/'];
    gtPath = [p.GTDir curSetName '/GT/'];
    
    imgFiles = dir([salPath '*.png']);
    imgNUM = length(imgFiles);

    %evaluation score initilization.
    Smeasure=zeros(1,imgNUM);
    Fmeasure=zeros(1,imgNUM);
    threshold_Fmeasure  = zeros(imgNUM,length(Thresholds));
    threshold_Emeasure  = zeros(imgNUM,length(Thresholds));
    MAE=zeros(1,imgNUM);

    tic;
    for i = 1:imgNUM
        fprintf('%d on %s \n', i, curSetName);
       %fprintf('Evaluating: %d/%d\n',i,imgNUM);

        name =  imgFiles(i).name;

        %load gt
        gt = imread([gtPath name(1:end-4) p.GTsuffix{curSet}]);


        if numel(size(gt))>2
            gt = rgb2gray(gt);
        end
        if ~islogical(gt)
            gt = gt(:,:,1) > 128;
        end

        %load salency
        sal  = imread([salPath name]);

        %check size
        if size(sal, 1) ~= size(gt, 1) || size(sal, 2) ~= size(gt, 2)
            sal = imresize(sal,size(gt));
            imwrite(sal,[salPath name]);
            fprintf('Error occurs in the path: %s!!!\n', [salPath name]);

        end

        sal = im2double(sal(:,:,1));

        %normalize sal to [0, 1]
        sal = reshape(mapminmax(sal(:)',0,1),size(sal));
        Smeasure(i) = StructureMeasure(sal,logical(gt));
                
%                  Using the 2 times of average of sal map as the threshold. 
        threshold =  2* mean(sal(:)) ;
        temp = Fmeasure_calu(sal,double(gt),size(gt),threshold);
        Fmeasure(i) = temp(3);

        for t = 1:length(Thresholds)
            threshold = Thresholds(t);
            temp = Fmeasure_calu(sal,double(gt),size(gt),threshold);
            threshold_Fmeasure(i,t) = temp(3);
            temp = Emeasure_calu(sal,double(gt),size(gt),threshold);
            threshold_Emeasure(i,t) = temp;
        end

        MAE(i) = mean2(abs(double(logical(gt)) - sal));

    end

    toc;
    f_total{curSet} = threshold_Fmeasure;
    e_total{curSet} = threshold_Emeasure;

    Sm = mean2(Smeasure);
    smeasure_total(curSet) = Sm;
    column_F = mean(threshold_Fmeasure,1);
    meanF = mean(column_F);
    meanF_total(curSet) = meanF;
    maxF = max(column_F);
    maxF_total(curSet) = maxF;

    column_E = mean(threshold_Emeasure,1);
    meanE = mean(column_E);
    meanE_total(curSet) = meanE;
    maxE = max(column_E);
    maxE_total(curSet) = maxE;

    mae = mean2(MAE);
    mae_total(curSet) = mae;
end
measure = [smeasure_total, meanF_total, meanE_total, mae_total]


