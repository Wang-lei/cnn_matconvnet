function jittered_im = image_jittering(im,netnorm,varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.border = [29, 29] ;
opts.augmentation = 'f25' ;
opts.interpolation = 'bilinear' ;
opts.numAugments = 1 ;
opts.keepAspect = true;
opts.isopticalflow = false; % if it's optical flow, R chanel (horizontal movements) should be reversed after flip
opts = vl_argparse(opts, varargin);

switch opts.augmentation
    case 'f5'
        tfs = [...
            .5 0 0 1 1 .5 0 0 1 1 ;
            .5 0 1 0 1 .5 0 1 0 1 ;
            0 0 0 0 0  1 1 1 1 1] ;
    case 'f25'
        [tx,ty] = meshgrid(linspace(0,1,5)) ;
        tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
        tfs_ = tfs ;
        tfs_(3,:) = 1 ;
        tfs = [tfs,tfs_] ;
    otherwise
        error('Unknown jittering method')
end

nbimages = size(im,4);

jittered_im = zeros(netnorm.imageSize(1), netnorm.imageSize(2), 3, ...
    nbimages*opts.numAugments, 'single') ;

[~,augmentations] = sort(rand(size(tfs,2), nbimages), 1) ;

si = 1 ;
for i=1:nbimages
    imt = im(:,:,:,i) ;
    
    % resize
    w = size(imt,2) ;
    h = size(imt,1) ;
    factor = [(netnorm.imageSize(1)+opts.border(1))/h ...
        (netnorm.imageSize(2)+opts.border(2))/w];
    
    if opts.keepAspect
        factor = max(factor) ;
    end
    if any(abs(factor - 1) > 0.0001)
        imt = imresize(imt, ...
            'scale', factor, ...
            'method', opts.interpolation) ;
    end
    
    % crop & flip
    w = size(imt,2) ;
    h = size(imt,1) ;
    for ai = 1:opts.numAugments
        t = augmentations(ai,i) ;
        tf = tfs(:,t) ;
        dx = floor((w - netnorm.imageSize(2)) * tf(2)) ;
        dy = floor((h - netnorm.imageSize(1)) * tf(1)) ;
        sx = (1:netnorm.imageSize(2)) + dx ;
        sy = (1:netnorm.imageSize(1)) + dy ;
        if tf(3), % flip
            sx = fliplr(sx) ;
            if opts.isopticalflow
                imt(sy,sx,1)=255-imt(sy,sx,1); % inverse horizontal flow
            end
        end
        jittered_im(:,:,:,si) = imt(sy,sx,:) ;
        
        si = si + 1 ;
    end
end
end
