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
cmd:option('-valid_set_size', 100) -- we use different data for test and validation       
cmd:option('-test_set_size', 342*10)       
cmd:option('-test_result_fname', 'test-contrast-max')
cmd:option('-test_err_th', 3)

-- feature network parameters
cmd:option('-net_fname', 'work/contrast-max/fnet_2016_09_29_12:57:49_contrast-max.t7')
cmd:option('-net_nb_feature', 64)
cmd:option('-net_kernel', 3)
cmd:option('-net_nb_layers', 4)

prm = cmd:parse(arg)
prm['arch'] = arch

-- make output folder
paths.mkdir('work/'..prm['test_result_fname']); -- make output folder
print('Testing net ' .. prm['net_fname'] .. ' arhitecture \n')

-- |load modules|

-- standard modules
require 'gnuplot'
require 'optim'
require 'nn'
require 'image'

-- custom modules
dofile('CAddMatrix.lua')                  -- Module that adds constant matrix to the input (I use it for masking purposes)

require 'libdprog'                        -- C++ module for dynamic programming
dofile('CContrastDprog.lua');             -- Contrastive dynamic programming module
dofile('CContrastMax.lua');               -- Contrastive max-2ndMax module
dofile('CMilDprog.lua');
dofile('CMilContrastDprog.lua')
dofile('DataLoader.lua');                 -- Parent class for dataloaders
dofile('CUnsup3EpiSet.lua');              -- Unsupervised training set loader
dofile('CSup2EpiSet.lua');          -- Supervised validation set loader

baseNet = dofile('CBaseNet.lua');         -- Function that makes base net
netWrapper = dofile('CNetWrapper.lua')    -- Function that "wrap" base net into training net
testFun = dofile('CTestUtils.lua');         -- Function that performs test on validation set

utils = dofile('utils.lua');              -- Utils for loading and visualization

-- cuda modules
require 'cunn'
require 'cutorch'
require 'cudnn'

-- mc-cnn
package.cpath = package.cpath .. ';../mc-cnn/?.so'
require('libcv')
include('../mc-cnn/Margin2.lua') 
include('../mc-cnn/Normalize2.lua') 
include('../mc-cnn/BCECriterion2.lua')

include('../mc-cnn/StereoJoin.lua') 
include('../mc-cnn/StereoJoin1.lua') 
include('../mc-cnn/SpatialConvolution1_fw.lua') 
include('../mc-cnn/SpatialLogSoftMax.lua') 

print('Parameters of the procedure : \n')
utils.printTable(prm)

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
    _BASE_FNET_ = utils.copynet(_BASE_FNET_, net)
else
  error('Network is not found\n')
end

-- |define datasets|
local supSet = sup2EpiSet(img1_arr[{{1,194},{},{}}], img2_arr[{{1,194},{},{}}], disp_arr[{{1,194},{},{}}], hpatch);
supSet:shuffle()  -- shuffle to have patches from all images

-- get test set
-- test set follows validation set in shuffled set.. since we fix random seed position of all examples is same as during training
test_set_start = (prm['valid_set_size']) + 1;
test_set_end = (prm['test_set_size']) + (prm['valid_set_size']);
_TE_INPUT_, _TE_TARGET_ = supSet:index(torch.range(test_set_start, test_set_end))

-- put validation set on gpu if needed  
_TE_TARGET_ = _TE_TARGET_:cuda()
_TE_INPUT_[1] = _TE_INPUT_[1]:cuda();
_TE_INPUT_[2] = _TE_INPUT_[2]:cuda();

-- |test|
local distNet = netWrapper.getDistNet(img_w, disp_max, hpatch, _BASE_FNET_:clone():double())
distNet:cuda()
local test_acc_lt3, errCases = testFun.getTestAcc(distNet, _TE_INPUT_, _TE_TARGET_, prm['test_err_th'])

-- |save test report|
-- save test parameters
timestamp = os.date("%Y_%m_%d_%X_")
torch.save('work/' .. prm['test_result_fname'] .. '/params_' .. timestamp .. prm['test_result_fname'] .. '.t7', prm, 'ascii');

-- save errorneous test samples
local fail_img = utils.vis_errors(errCases[1], errCases[2], errCases[3], errCases[4])
image.save('work/' .. prm['test_result_fname'] .. '/error_cases_' .. timestamp .. prm['test_result_fname'] .. '.png',fail_img)

-- save distance matrices
local lines = {3, 9, 14, 53}
for nline = 1,#lines do
    
    local distMat, gtDistMat = testFun.getDist(distNet, 
                              {_TE_INPUT_[1][{{lines[nline]},{},{}}], 
                               _TE_INPUT_[2][{{lines[nline]},{},{}}]}, 
                               _TE_TARGET_[{{lines[nline]},{},{}}], prm['test_err_th'])

    local gtDistMat = -utils.scale2_01(gtDistMat)+1

    local distMat = utils.softmax(distMat:squeeze())
    local distMat = -utils.scale2_01(distMat)+1

    local r = distMat:clone()
    local g = distMat:clone()
    local b = distMat:clone()

    r[gtDistMat:eq(0)] = 0 
    g[gtDistMat:eq(0)] = 1
    b[gtDistMat:eq(0)] = 0

    im = torch.cat({nn.utils.addSingletonDimension(r,1), nn.utils.addSingletonDimension(g,1), nn.utils.addSingletonDimension(b,1)}, 1)
   
   image.save('work/' .. prm['test_result_fname'] .. '/dist_' ..  string.format("line%i_",lines[nline])  .. timestamp .. prm['test_result_fname'] .. '.png',im)
  end

-- save log
logger = optim.Logger('work/' .. prm['test_result_fname'] .. '/'.. prm['test_result_fname'], true)
logger:setNames{'Accuracy (<3 disparity err)'}
logger:add{test_acc_lt3}






