require 'gnuplot'
require 'image'
require 'torch'
require 'lfs'
require 'os'

 torch.manualSeed(1)

utils = dofile('utils.lua');              
dofile('CUnsupPipeKITTI.lua');

hpatch = 4
batch_size = 128
set = 'kitti_ext'
net_fname = '/HDD1/Dropbox/Research/01_code/mil-mc-cnn/work/TRAIN_CONTRASTIVEDP_FSTXXL_KITTIEXT/metricNet_TRAIN_CONTRASTIVEDP_FSTXXL_KITTIEXT.t7'

unsupSet = unsupPipeKITTI('data/kitti_ext', set, hpatch);
_TR_INPUT_, _WIDTH_TAB_, _DISP_MAX_TAB_  = unsupSet:get( batch_size, net_fname )



