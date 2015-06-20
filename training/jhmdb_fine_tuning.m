function jhmdb_fine_tuning

matconvpath = '/sequoia/data1/gcheron/code/cnn_matconvnet/matconvnet-1.0-beta11';
run([matconvpath '/matlab/vl_setupnn.m']) ;
addpath([matconvpath '/examples'],'/sequoia/data1/gcheron/code/cnn_matconvnet/training/datasets_code/JHMDB','/sequoia/data1/gcheron/code/cnn_matconvnet/training/training_code')

%all_parts={'hands','full_body','full_image','upper_body','left_hand','right_hand'};

gpuID = 1 ;

for split = 1:1

    % batch opt
    opts.batchwrapper.numFetchThreads = 6 ;
    opts.batchwrapper.do_jittering=1;
    opts.batchwrapper.opts_jittering.isopticalflow=0; %1
    opts.batchwrapper.do_background_proposals = 1 ;
    opts.batchwrapper.opts_background_proposals.box_id = 4 ; %[1 lhand, 2 rhand, 3 upbody, 4 fullbody]
    
    opts.part = 'full_body' ;
    %opts.part ='left_hand' ;
    
    opts.split = split ;
    opts.splitpath = '/sequoia/data1/gcheron/ICCV15/JHMDB/splitlists/experimentsplits' ;
    opts.dataDir = sprintf('/tmp/JHMDB/cnn_images/%s/resized',opts.part) ; %sprintf('/tmp/JHMDB/cnn_OF/%s/resized227',opts.part) ; sprintf('/home/local/gcheron/datasets/JHMDB/cnn_OF/%s/resized227',opts.part) ;
    opts.originalProposalDataDir = '/tmp/JHMDB/cnn_images/full_image'; % /tmp/JHMDB/cnn_OF/full_image %'/home/local/gcheron/datasets/JHMDB/cnn_OF/full_image';
    
    %opts.expDir = sprintf('/sequoia/data1/gcheron/code/cnn_matconvnet/training/experiments/JHMDB_fine-tuning/JHMDB_split%d_%s',opts.split,opts.part) ;
    %opts.expDir = sprintf('/home/local/gcheron/cnn_matconvnet/JHMDB/experiments/JHMDB_fine-tuning/JHMDB_split%d_%s',opts.split,opts.part) ;
    %opts.expDir = sprintf('/home/local/gcheron/cnn_matconvnet/JHMDB/experiments/JHMDB_fine-tuning/JHMDB_split%d_%s_BPfc1conv_WBsameLR',opts.split,opts.part) ;
    opts.expDir = sprintf('/home/local/gcheron/cnn_matconvnet/JHMDB/experiments/JHMDB_fine-tuning/JHMDB_split%d_%s_APP_BPall_WBsameLR',opts.split,opts.part) ;
   
    opts.actionsnames={'brush_hair','catch','clap','climb_stairs','golf','jump','kick_ball','pick','pour','pullup','push','run','shoot_ball','shoot_bow','shoot_gun','sit','stand','swing_baseball','throw','walk','wave'} ;
    opts.joitsDir='/sequoia/data1/gcheron/JHMDB/joint_positions';
    
    opts.backgroundnum=200000 ; % 55000;
    %opts.imdbPath = sprintf('/sequoia/data1/gcheron/code/cnn_matconvnet/training/experiments/JHMDB_fine-tuning/imdb_JHMDB_split%d_%s_bnum%i.mat',opts.split,opts.part,opts.backgroundnum);
    opts.imdbPath = sprintf('/sequoia/data1/gcheron/code/cnn_matconvnet/training/experiments/JHMDB_fine-tuning/imdb_JHMDB_APP_split%d_%s_bnum%i.mat',opts.split,opts.part,opts.backgroundnum);
    opts.lite = false ;
    
    opts.netpath = 'imagenet-vgg-f.mat';%'bignet_flow_wmag_finetune_split1_iter_50000.mat';
    opts.train.prefetch = true ;
    opts.train.batchSize = 128 ;
    opts.train.numEpochs = 60 ;
    opts.train.continue = true ;
    opts.train.gpus = gpuID ;
    opts.train.learningRate = [1e-3*ones(1,20) 1e-4*ones(1,20) 1e-5*ones(1,20)] ;
    opts.train.expDir = opts.expDir ;
    opts.train.backPropDepth = +inf ; % 11 % until conv5 % +inf; % BP all ; %8 all FC ;
    
    % -------------------------------------------------------------------------
    %                                                    Network initialization
    % -------------------------------------------------------------------------
    net = initializeNetwork(opts) ;
    opts.batchwrapper.net_normalization=net.normalization;
    % -------------------------------------------------------------------------
    %                                                   Database initialization
    % -------------------------------------------------------------------------
    fprintf('PART: %s | SPLIT: %d\n',opts.part,opts.split);
    fprintf('IMDIR: %s\n',opts.dataDir);
    
    if exist(opts.imdbPath,'file')
        fprintf('LOAD: %s\n',opts.imdbPath);
        imdb = load(opts.imdbPath) ;
        if isfield(imdb,'imdb') ; imdb = imdb.imdb ; end
    else
        %imdb = cnn_JHMDB_setup_data_VALonTEST(opts) ;
        imdb = cnn_JHMDB_setup_data_with_background_VALonTEST(opts) ;
        mkdir(opts.expDir) ;
        save(opts.imdbPath, '-struct', 'imdb') ;
    end
    
    
    
    % -------------------------------------------------------------------------
    %                                               Stochastic gradient descent
    % -------------------------------------------------------------------------
    
    fn = getBatchWrapper(opts.batchwrapper) ;
    
    [net,info] = cnn_train(net, imdb, fn, opts.train, 'conserveMemory', true) ;   
end


% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts_batchwrapper)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts_batchwrapper) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatch(imdb, batch, opts_batchwrapper)
% -------------------------------------------------------------------------

bs=length(batch); % batch size

% get image names
images = imdb.images.name(batch) ;

% if prefetch
if nargout == 0
    % load images
    vl_imreadjpeg(images, 'numThreads', opts_batchwrapper.numFetchThreads, 'prefetch') ;
    im = [] ;labels = [] ;
    return ;
end

% load images
images = vl_imreadjpeg(images,'numThreads', opts_batchwrapper.numFetchThreads) ;


% train or validation?
isval=imdb.images.set(batch(1))==2;

% background proposals
if opts_batchwrapper.do_background_proposals && ~isval
    boxes=imdb.images.bbox(batch);
    doproposal = ~cellfun(@isempty,boxes); % compute background proposals on those samples
    
    box_id = opts_batchwrapper.opts_background_proposals.box_id; % get the box position in the box list
    
    % normlization info
    sh=opts_batchwrapper.net_normalization.imageSize(1);
    sw=opts_batchwrapper.net_normalization.imageSize(2);
    metinterpolation=opts_batchwrapper.net_normalization.interpolation;
    
    % init images
    im=single(zeros(sh,sw,3,bs));
    
    parfor ii=1:bs
        if ~doproposal(ii)
            im(:,:,:,ii)=images{ii};
            continue;
        end
        box=boxes{ii};
        patch = extract_background_patch(images{ii},box(box_id,:)) ; % get background proposals
        
        im(:,:,:,ii)= imresize(patch, [sh,sw], 'method', metinterpolation) ;
    end
else
    im = cat(4,images{:}) ;
    doproposal=false(bs,1); % no background proposal
end

% jittering
if opts_batchwrapper.do_jittering && ~isval
    im(:,:,:,~doproposal) = image_jittering(im(:,:,:,~doproposal),opts_batchwrapper.net_normalization,opts_batchwrapper.opts_jittering);
end

% image normalization
im = bsxfun(@minus, im, opts_batchwrapper.net_normalization.averageImage) ;

% get labels
labels = imdb.images.label(batch) ;

% -------------------------------------------------------------------------
function net = initializeNetwork(opt)
% -------------------------------------------------------------------------

net=load(opt.netpath);
for i=1:length(net.layers)
    if ~strcmp(net.layers{i}.type,'conv') ; continue ; end
    %net.layers{i}.learningRate = [1 2] ;
    net.layers{i}.learningRate = [1 1] ;
    net.layers{i}.weightDecay = [1 0] ;
end
net.layers{end-1} = struct('type', 'conv', ...
    'weights', [], ...
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', [1 1], ...    %'learningRate', [10 20], ...
    'weightDecay', [1 0], ...
    'name', 'fc8_JHMDB');
%net.layers{end-1}.weights ={0.01 * randn(1,1,4096,21,'single'),zeros(1, 21, 'single')};
net.layers{end-1}.weights ={0.01 * randn(1,1,4096,22,'single'),zeros(1, 22, 'single')};

net.classes = opt.actionsnames ;

% insert 2 drop out layers after the 2 first fully connected layers: 21 => 23 layers
net.layers{end+1} = [] ;
net.layers{end+1} = [] ;

net.layers{end}=net.layers{end-2} ; % softmaxloss 23
net.layers{end-1}=net.layers{end-3} ; % fine-tuned layer 22
net.layers{end-2}=struct('type', 'dropout','rate', 0.5) ; % drop out 21
net.layers{end-3}=net.layers{end-4} ; % relu 20
net.layers{end-4}=net.layers{end-5} ; % 2nd fully 19
net.layers{end-5}=struct('type', 'dropout','rate', 0.5) ; % drop out 18


