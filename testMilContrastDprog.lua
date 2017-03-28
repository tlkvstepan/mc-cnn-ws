require 'gnuplot'
require 'optim'
require 'nn'
require 'cunn'

-- Custom modules
dofile('CAddMatrix.lua')                  -- Module that adds constant matrix to the input (I use it for masking purposes)

require 'libdprog'                        -- C++ module for dynamic programming              
dofile('CMilContrastDprog.lua');             -- Contrastive dynamic programming module

dofile('DataLoader.lua');                 -- Parent class for dataloaders
dofile('CUnsup3EpiSet.lua');              -- Unsupervised training set loader
dofile('CSup2EpiSet.lua');          -- Supervised validation set loader

baseNet = dofile('CBaseNet.lua');         -- Function that makes base net
netWrapper = dofile('CNetWrapper.lua')    -- Function that "wrap" base net into training net
testFun = dofile('CTestUtils.lua');         -- Function that performs test on validation set

utils = dofile('utils.lua');              -- Utils for loading and visualization

local loss_margin = 0.2
local dist_min = 2
local occ_th = 8   
local dist_min = 1

-- feature network parameters
local net_nb_feature = 64
local net_kernel = 3
local net_nb_layers = 4

math.randomseed(0); 
torch.manualSeed(0)

-- |read trainng data| (KITTI)
local img1_arr = torch.squeeze(utils.fromfile('data/KITTI12/x0.bin'));
local img2_arr = torch.squeeze(utils.fromfile('data/KITTI12/x1.bin'));
local disp_arr = torch.round(torch.squeeze(utils.fromfile('data/KITTI12/dispnoc.bin')));

local disp_max = disp_arr:max()
local img_w = img1_arr:size(3);

_BASE_FNET_, hpatch = baseNet.get(net_nb_layers, net_nb_feature, net_kernel)
_TR_NET_, _CRITERION_ = netWrapper.getMilContrastDprog(img_w, disp_max, hpatch, dist_min, occ_th, loss_margin, _BASE_FNET_)
_BASE_PPARAM_ = _BASE_FNET_:getParameters() 
_TR_PPARAM_, _TR_PGRAD_ = _TR_NET_:getParameters()

local unsupSet = unsup3EpiSet(img1_arr, img2_arr, hpatch, disp_max);

-- get validation set from supervised set
_TR_NET_:cuda()
_CRITERION_:cuda()
start_time = os.time()
_TR_ERR_ = 0
for i = 1,100 do
  input, target = unsupSet:index(torch.Tensor{i})
  input[1] = input[1]:cuda()
  input[2] = input[2]:cuda()
  input[3] = input[3]:cuda()
  output = _TR_NET_:forward({input[1], input[2], input[3]})
  local   nb_tables = #_TR_NET_.output

  -- if nuber of nonempty output tables is 0, we can not do anything
  if nb_tables ~= 0 then

    -- make target array for every table, and simultaneously compute 
    -- total number of samples
    local sample_target = {};
    for ntable = 1,nb_tables do
      local nb_comp = _TR_NET_.output[ntable][1]:numel()
      sample_target[ntable] = nn.utils.addSingletonDimension(_TR_NET_.output[ntable][1]:clone():fill(1),1) 
    end

    -- pass through criterion
    _TR_ERR_ = _TR_ERR_ + _CRITERION_:forward(_TR_NET_.output, sample_target)

    -- backword pass
    _TR_NET_:backward(input, _CRITERION_:backward(_TR_NET_.output, sample_target))
    collectgarbage()

  end
   
 end
end_time = os.time()
print(os.difftime(end_time, start_time))
