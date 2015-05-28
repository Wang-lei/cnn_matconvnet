function testGPU(varargin)

run(fullfile('/sequoia/data1/gcheron/code/cnn_matconvnet/matconvnet-1.0-beta11', ...
    'matlab', 'vl_setupnn.m')) ;

opts.dataDir = [] ;
opts.train.batchSize = 1;
opts.train.numEpochs = 100 ;
opts.train.continue = false ;
opts.train.conserveMemory = true;
opts.train.useGpu = true ;
opts.train.learningRate = 0.01;
opts.train.expDir = [];
opts = vl_argparse(opts, varargin) ;

% get the network
 net = initializeNetwork_AlexNet_tuanHung;
imageSize = [227, 227];
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

% Take the mean out and make GPU if needed
numImages = 8192;
rng(1,'twister');
imdb.images.data = single( rand(imageSize(1), imageSize(2), 3, 1) );
if opts.train.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

imdb.images.set = ones(numImages, 1);
imdb.images.set( round(numImages / 2) + 1 : end) = 3; 
[net,info] = cnn_train_testOnly(net, imdb, @getBatch, ...
    opts.train, ...
    'val', [] );
end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,ones(length(batch), 1));
labels = ones(1, length(batch) );
end
