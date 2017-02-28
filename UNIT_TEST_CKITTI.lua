require 'gnuplot'
require 'image'
require 'torch'
require 'lfs'
require 'os'

 torch.manualSeed(1)

utils = dofile('utils.lua');              
dofile('CKITTI.lua');

hpatch = 4
batch_size = 128
set = 'kitti_ext'

unsupSet = kitti('/home/tulyakov/Desktop/kitti_ext', set, hpatch);
_TR_INPUT_, _WIDTH_TAB_, _DISP_MAX_TAB_  = unsupSet:get( batch_size, net_fname )



