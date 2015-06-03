function extract_JHMDB_features()

% get list of images
[filelist,outlist]=get_OF_list();

% load net
gpuDevice(1);
matconvpath = '/sequoia/data1/gcheron/code/cnn_matconvnet/matconvnet-1.0-beta11';
run([matconvpath '/matlab/vl_setupnn.m']) ;
%netpath='/sequoia/data1/gcheron/code/cnn_matconvnet/training/experiments/JHMDB_fine-tuning/JHMDB_split1_full_body_BPall/net-epoch-60.mat';laynum=20;%21
netpath='/sequoia/data1/gcheron/code/cnn_matconvnet/import_caffe_networks/OF_caffe_net/bignet_flow_wmag_finetune_split1_iter_50000.mat';laynum=19;
net=load(netpath);net=net.net; net.layers = net.layers(1:end-1);
net = vl_simplenn_move(net, 'gpu') ;
bsize=128;
nim=length(filelist);
for i=1:bsize:nim
    
    fprintf('feature extraction: %d over %d:   ',i,nim);tic;
    im = vl_imreadjpeg(filelist(i:min(i+bsize-1,nim)),'numThreads', 20) ;
    im = cat(4,im{:}) ;
    im = bsxfun(@minus, im, net.normalization.averageImage) ;
    im = gpuArray(im) ;
    res=vl_simplenn(net,im,[],[],'disableDropout',1);
    fprintf('extract %fs   ',toc);tic;
    save_feats(squeeze(res(laynum).x),outlist(i:min(i+bsize-1,nim))); % res(i+1).x: the output of layer i
    fprintf('save %fs\n',toc)
end

function [filelist,outlist]=get_OF_list()
bodyparts={'full_body','full_image','left_hand','right_hand','upper_body'};
flowdir='/tmp/JHMDB/cnn_OF' ;
%outdir='/sequoia/data1/gcheron/JHMDB/cnn_OF_finefullbody_split1/features';
%outdir='/sequoia/data1/gcheron/JHMDB/cnn_OF_finefullbodynorelu_split1/features';
outdir='/sequoia/data1/gcheron/JHMDB/cnn_OF_norelu/features';

filelist={};
outlist={};
for i=1:length(bodyparts)
    f = fopen([flowdir '/OF_jpg_list.txt'], 'r');
    partpath=[outdir '/' bodyparts{i}];
    mkdir(partpath);
    
    line=fgetl(f);
    while(ischar(line))
        curfile=[flowdir '/'  bodyparts{i} '/resized227/' line];
        [~,we,~]=fileparts(line);
        curout=[partpath '/' we '.mat'];
        filelist{end+1}=(curfile);
        outlist{end+1}=(curout);
        line=fgetl(f);
    end
    fclose(f);
    
end

function save_feats(feats,outlist)
assert(length(outlist)==size(feats,2));

parfor i=1:length(outlist)
    out=outlist{i};
    features=gather(feats(:,i))';
    parsave(out,features);
end


function parsave(out,features)
save(out,'features');






