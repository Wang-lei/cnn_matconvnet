#! /bin/bash
pushd `dirname $0` > /dev/null
popd > /dev/null

echo "PYTHONPATH:  $PYTHONPATH"

#python my_caffe_matconvnet_wrapper.py --caffe-variant=caffe --preproc=caffe     --average-value="(128.0, 128.0, 128.0)"     "/home/gcheron/software/import_caffe_networks/OF_caffe_net/superhuman_train_ucfsports.prototxt" "/home/gcheron/software/import_caffe_networks/OF_caffe_net/pascal_finetune_hyb2_VOC2012_train_iter_25000"     "/home/gcheron/software/import_caffe_networks/OF_caffe_net/pascal_finetune_hyb2_VOC2012_train_iter_25000.mat"
python my_caffe_matconvnet_wrapper.py --caffe-variant=caffe --preproc=caffe     --average-value="(128.0, 128.0, 128.0)"     "/home/gcheron/software/import_caffe_networks/OF_caffe_net/superhuman_train_ucfsports.prototxt" "/home/gcheron/software/import_caffe_networks/OF_caffe_net/bignet_flow_wmag_finetune_split1_iter_50000"     "/home/gcheron/software/import_caffe_networks/OF_caffe_net/bignet_flow_wmag_finetune_split1_iter_50000.mat"
