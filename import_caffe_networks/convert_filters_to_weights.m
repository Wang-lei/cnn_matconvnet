netpath='OF_caffe_net/bignet_flow_wmag_finetune_split1_iter_50000.mat' ; %'OF_caffe_net/pascal_finetune_hyb2_VOC2012_train_iter_25000.mat'
net=load(netpath);
if isfield(net,'net'), net=net.net ; end
for i=1:length(net.layers)
	if strcmp(net.layers{i}.type,'conv')
		net.layers{i}.weights{1}=net.layers{i}.filters ;
		net.layers{i}.weights{2}=net.layers{i}.biases; 
		net.layers{i}=rmfield(net.layers{i},'filters');
		net.layers{i}=rmfield(net.layers{i},'biases');
	end
	if strcmp(net.layers{i}.type,'softmax_loss')
		net.layers{i}.type = 'softmaxloss';
	end
end

save(netpath,'net');
