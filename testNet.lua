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

mcCnnFst = dofile('CMcCnnFst.lua');
dofile('CAddMatrix.lua')
milWrapper = dofile('CMilWrapper.lua')
utils = dofile('utils.lua');

fnet0 = torch.load('../mc-cnn/net/net_kitti_fast_-a_train_all.t7', 'ascii')[1]
-- remove normalization and stereo
fnet0:remove(9)
fnet0:remove(8)

--fnet0_param = fnet0:getParameters()

fnet = mcCnnFst.get(4,64,3)
--fnet_param = fnet:getParameters()

fnet = utils.copynet(fnet, fnet0)

inp = torch.rand(1,9,9)
out1 = fnet:forward(inp)
out2 = fnet0:forward(inp:cuda())
out2 = out2:double()

print(torch.all(torch.eq(torch.squeeze(out1),torch.squeeze(out2[{{},{5},{5}}]))))

print(torch.squeeze(out1))
print(torch.squeeze(out2[{{},{5},{5}}]))

