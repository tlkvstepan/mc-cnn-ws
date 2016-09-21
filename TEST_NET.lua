--[[ 
This is universal script for testing of network 
It supports or and mc-cnn networks
]]--

require 'torch'

-- |read input parameters|
-- fist argument is net architecture: our or mc-cnn
arch = table.remove(arg, 1) 
assert(arch == 'our' or arch == 'mc-cnn')

cmd = torch.CmdLine()

-- test parameters parameters
cmd:option('-valid_set_size', 50000) -- we use different data for test and validation       
cmd:option('-test_set_size', 10000)       
cmd:option('-test_result_fname', 'test')

-- feature network parameters
cmd:option('-net_fname', '')
cmd:option('-net_nb_feature', 64)
cmd:option('-net_kernel', 3)
cmd:option('-net_nb_layers', 4)

-- debug
cmd:option('-debug_gpu_on', true)

prm = cmd:parse(arg)
prm['arch'] = arch


-- make output folder
paths.mkdir('work/'..prm['test_result_fname']); -- make output folder
print('Testing net ' .. prm['net_fname'] .. ' arhitecture \n')
  
-- |load modules|

-- standard modules
require 'gnuplot'
require 'nn'
require 'image'
require 'optim'
package.cpath = package.cpath .. ';../mc-cnn/?.so'
require('libcv')

-- mc-cnn modules
include('../mc-cnn/Margin2.lua')
include('../mc-cnn/Normalize2.lua')
include('../mc-cnn/BCECriterion2.lua')
include('../mc-cnn/StereoJoin.lua')
include('../mc-cnn/StereoJoin1.lua')
include('../mc-cnn/SpatialConvolution1_fw.lua')
include('../mc-cnn/SpatialLogSoftMax.lua')

-- custom modules for our network
dofile('CAddMatrix.lua')                  -- Module that adds constant matrix to the input (I use it for masking purposes)

require 'libdprog'                        -- C++ module for dynamic programming
dofile('CDprog.lua');                     -- Dynamic programming module
dofile('CContrastDprog.lua');             -- Contrastive dynamic programming module
dofile('CContrastMax.lua');               -- Contrastive max-2ndMax module

dofile('DataLoader.lua');                 -- Parent class for dataloaders
dofile('CUnsup3EpiSet.lua');              -- Unsupervised training set loader
dofile('CSup1Patch1EpiSet.lua');          -- Supervised validation set loader

baseNet = dofile('CBaseNet.lua');         -- Function that makes base net
netWrapper = dofile('CNetWrapper.lua')    -- Function that "wrap" base net into training net
testFun = dofile('CTestFun.lua');         -- Function that performs test on validation set

utils = dofile('utils.lua');              -- Utils for loading and visualization

print('Parameters of the procedure : \n')
utils.printTable(prm)

if( not prm['debug_gpu_on'] and arch == 'mc-cnn' ) then
   error('Test for mc-cnn network is only possible with gpu')
end  

if( prm['debug_gpu_on'] ) then            
  require 'cunn'
  require 'cunn'
  require 'cutorch'
  require 'libadcensus'
  require 'cudnn'
end

-- Set random seeds for math and torch for repeatability
math.randomseed(0); 
torch.manualSeed(0)

-- |read test data| (KITTI)
local img1_arr = torch.squeeze(utils.fromfile('data/KITTI12/x0.bin'));
local img2_arr = torch.squeeze(utils.fromfile('data/KITTI12/x1.bin'));
local disp_arr = torch.round(torch.squeeze(utils.fromfile('data/KITTI12/dispnoc.bin')));

local disp_max = disp_arr:max()
local img_w = img1_arr:size(3);

-- |load network|
_BASE_FNET_, hpatch = baseNet.get(prm['net_nb_layers'], prm['net_nb_feature'], prm['net_kernel'])
if utils.file_exists(prm['net_fname']) then
  if arch == 'mc-cnn' then
    net = torch.load(prm['net_fname'], 'ascii')[1]
    -- remove normalization and stereo layers
    net:remove(9)
    net:remove(8)
  else
    net = torch.load(prm['net_fname'], 'ascii')
  end
    -- since mc-cnn contains cudnn modules
    -- we can not copy parameters vector directly
    _BASE_FNET_ = utils.copynet(_BASE_FNET_,net)
else
  error('Network is not found\n')
end

-- put base neton gpu if needed
if prm['debug_gpu_on'] then
  _BASE_FNET_:cuda()
end

-- |define datasets|
local unsupSet = unsup3EpiSet(img1_arr, img2_arr, hpatch, disp_max);
local supSet = sup1Patch1EpiSet(img1_arr[{{1,194},{},{}}], img2_arr[{{1,194},{},{}}], disp_arr[{{1,194},{},{}}], hpatch);
supSet:shuffle()  -- shuffle to have patches from all images

-- get test set
-- test set follows validation set in shuffled set.. since we fix random seed position of all examples is same as during training
test_set_start = (prm['valid_set_size']) + 1;
test_set_end = (prm['test_set_size']) + (prm['valid_set_size']);
_TE_INPUT_, _TE_TARGET_ = supSet:index(torch.range(test_set_start, test_set_end))

-- put validation set on gpu if needed  
if prm['debug_gpu_on'] then
  _TE_TARGET_ = _TE_TARGET_:cuda()
  _TE_INPUT_[1] = _TE_INPUT_[1]:cuda();
  _TE_INPUT_[2] = _TE_INPUT_[2]:cuda();
end

-- |test|
local test_acc_lt3, test_acc_lt5, errCases = testFun.epiEval(_BASE_FNET_, _TE_INPUT_, _TE_TARGET_)

-- |save test report|
-- save test parameters
timestamp = os.date("%Y_%m_%d_%X_")
torch.save('work/' .. prm['test_result_fname'] .. '/params_' .. timestamp .. prm['test_result_fname'] .. '.t7', prm, 'ascii');

-- save errorneous test samples
local fail_img = utils.vis_errors(errCases[1], errCases[2], errCases[3], errCases[4])
image.save('work/' .. prm['test_result_fname'] .. '/error_cases_' .. timestamp .. prm['test_result_fname'] .. '.png',fail_img)

-- save distance matrices
 local lines = {290,433}
for nline = 1,#lines do
  local input = unsupSet:index(torch.Tensor{lines[nline]})
  local test_net = netWrapper.getDistNet(_BASE_FNET_)
  if  prm['debug_gpu_on'] then
    test_net:cuda()
    input[1] = input[1]:cuda()
    input[2] = input[2]:cuda();
    input[3] = input[3]:cuda();
  end
  local refPos = test_net:forward(input):float();
  refPos = utils.mask(refPos,disp_max)
  refPos = utils.softmax(refPos)
  refPos = utils.scale2_01(refPos)
  image.save('work/' .. prm['test_result_fname'] .. '/dist_' ..  string.format("line%i_",lines[nline])  .. timestamp .. prm['test_result_fname'] .. '.png',refPos)
end

-- save log
logger = optim.Logger('work/' .. prm['test_result_fname'] .. '/'.. prm['test_result_fname'], true)
logger:setNames{'Training loss', '<3 disparity error', '<5 disparity error'}
logger:add{train_err, test_acc_lt3, test_acc_lt5}






