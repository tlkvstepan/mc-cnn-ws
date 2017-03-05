require 'gnuplot'
require 'image'
require 'torch'
require 'lfs'
require 'os'

 torch.manualSeed(1)

utils = dofile('utils.lua');              
dofile('CUnsupPipeKITTI_with_GT.lua');

hpatch = 4
batch_size = 128
set = 'kitti_ext'
net_fname = '/HDD1/Dropbox/Research/01_code/mil-mc-cnn/work/test-2017_03_04_13:57:36/metricNet_test-2017_03_04_13:57:36.t7'
use_gt = true;

unsupSet = unsupPipeKITTI('data/kitti_ext', set, use_gt, hpatch);
_TR_INPUT_, _WIDTH_TAB_, _DISP_MAX_TAB_  = unsupSet:get( batch_size, net_fname )



