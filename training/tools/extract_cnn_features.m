function extract_cnn_features(net,laynum,listinnames,listoutnames,batchsize)
% compute CNN features from layler laynum from the list listinnames and
% save them to listoutnames (batch size is controled by batchsize)


nim=length(listinnames);
nsize=net.normalization.imageSize(1:2);
nmethod=net.normalization.interpolation;
for i=1:batchsize:nim
    fprintf('feature extraction: %d over %d:   ',i,nim);tic;
    imloaded = vl_imreadjpeg(listinnames(i:min(i+bsize-1,nim)),'numThreads', 20) ;
    im=zeros(nsize(1),nsize(2),3,length(imloaded));
    parfor j=1:length(imloaded)
        cim=imloaded{j};
        im(:,:,:,j) = imresize(cim, nsize,'method',nmethod) ;
    end
    fprintf('load and resize %fs   ',toc);tic;
    im = bsxfun(@minus, im, net.normalization.averageImage) ;
    im = gpuArray(im) ;
    res=vl_simplenn(net,im,[],[],'disableDropout',1);
    fprintf('extract %fs   ',toc);tic;
    save_feats(squeeze(res(laynum).x),listoutnames(i:min(i+bsize-1,nim)));
    fprintf('save %fs\n',toc)
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

