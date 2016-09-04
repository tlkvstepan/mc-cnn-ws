require 'cunn'
require 'cutorch'
require 'image'
require 'libadcensus'
require 'libcv'
require 'cudnn'

include('../mc-cnn/Margin2.lua')
include('../mc-cnn/Normalize2.lua')
include('../mc-cnn/BCECriterion2.lua')
include('../mc-cnn/StereoJoin.lua')
include('../mc-cnn/StereoJoin1.lua')
include('../mc-cnn/SpatialConvolution1_fw.lua')
include('../mc-cnn/SpatialLogSoftMax.lua')




fnet = torch.load('../mc-cnn/net/net_kitti_fast_-a_train_all.t7', 'ascii')[1]
-- remove normalization and stereo
fnet:remove(9)
fnet:remove(8)
print(fnet)
-- 

img_w = 100
disp_max = 10;
hpatch = 4;

_TR_NET_ = mcCnnFst.getMilNetDoubleBatch(img_w, disp_max, hpatch, fnet)
print(_TR_NET_)