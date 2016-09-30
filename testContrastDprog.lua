require 'gnuplot'
require 'optim'
require 'nn'

-- Custom modules
dofile('CAddMatrix.lua')                  -- Module that adds constant matrix to the input (I use it for masking purposes)

require 'libdprog'                        -- C++ module for dynamic programming
dofile('CDprog.lua');                     -- Dynamic programming module
dofile('CContrastDprog.lua');             -- Contrastive dynamic programming module
dofile('CContrastMax.lua');               -- Contrastive max-2ndMax module

dofile('DataLoader.lua');                 -- Parent class for dataloaders
dofile('CUnsup3EpiSet.lua');              -- Unsupervised training set loader
dofile('CSup2EpiSet.lua');          -- Supervised validation set loader

baseNet = dofile('CBaseNet.lua');         -- Function that makes base net
netWrapper = dofile('CNetWrapper.lua')    -- Function that "wrap" base net into training net
testFun = dofile('CTestUtils.lua');         -- Function that performs test on validation set

utils = dofile('utils.lua');              -- Utils for loading and visualization


-- |read trainng data| (KITTI)
local img1_arr = torch.squeeze(utils.fromfile('data/KITTI12/x0.bin'));
local img2_arr = torch.squeeze(utils.fromfile('data/KITTI12/x1.bin'));
local disp_arr = torch.round(torch.squeeze(utils.fromfile('data/KITTI12/dispnoc.bin')));

local disp_max = disp_arr:max()
local img_w = img1_arr:size(3);

_BASE_FNET_, hpatch = baseNet.get(4, 64, 3)
_TR_NET_, _CRITERION_ = netWrapper.getContrastDprog(img_w, disp_max, hpatch, 2, 0.2, _BASE_FNET_)
_BASE_PPARAM_ = _BASE_FNET_:getParameters() 
_TR_PPARAM_, _TR_PGRAD_ = _TR_NET_:getParameters()

local unsupSet = unsup3EpiSet(img1_arr, img2_arr, hpatch, disp_max);

-- get validation set from supervised set
input, target = unsupSet:index(torch.Tensor{10})


output = _TR_NET_:forward({input[1], input[2]})
local sample_traget = nn.utils.addSingletonDimension(_TR_NET_.output[1][1]:clone():fill(1),1)
_TR_ERR_ = _CRITERION_:forward(_TR_NET_.output, sample_traget)
_TR_NET_:backward(sample_input, _CRITERION_:backward(_TR_NET_.output, sample_traget))


gradOutput = torch.rand(output:size(1),output:size(2))
gradInput = module:backward(input, gradOutput)

print(out)
print(inputGrad)