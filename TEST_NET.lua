--[[ 
This is universal script for testing of network 
It supports or and mc-cnn networks
]]--

require 'torch'

-- |read input parameters|
-- fist argument is net architecture: our or mc-cnn
arch = table.remove(arg, 1) 
assert(arch == 'our' or arch == 'mc-cnn')

set = table.remove(arg, 1) 
assert(set == 'kitti' or set == 'kitti15' or set == 'mb')

cmd = torch.CmdLine()

-- test parameters parameters
cmd:option('-valid_set_size', 100) -- we use different data for test and validation       
cmd:option('-test_result_fname', 'test-contrast-max')
cmd:option('-test_err_th', 3)
cmd:option('-test_batch_size', 1024)

-- feature network parameters
cmd:option('-net_fname', 'work/contrast-dprog/fnet_2016_10_15_18:25:25_contrast-dprog.t7')
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

-- |read test data|
if set == 'kitti' then
  x0_fname = 'data/KITTI12/x0.bin'
  x1_fname = 'data/KITTI12/x1.bin'
  dispnoc_fname  = 'data/KITTI12/dispnoc.bin'
  nb_tr = 194
  nb_te = 195
elseif set == 'kitti15' then
  x0_fname = 'data/KITTI15/x0.bin'
  x1_fname = 'data/KITTI15/x1.bin'
  dispnoc_fname  = 'data/KITTI15/dispnoc.bin'
  nb_tr = 200
  nb_te = 200
elseif set == 'mb' then
end

local img1_arr = torch.squeeze(utils.fromfile(x0_fname));
local img2_arr = torch.squeeze(utils.fromfile(x1_fname));
local disp_arr = torch.round(torch.squeeze(utils.fromfile(dispnoc_fname)));
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
supSet.id = supSet.id[{{test_set_start, supSet:size()}}];   
--supSet.id = supSet.id[{{test_set_start, test_set_start+100}}];   

-- get network for test
local distNet = netWrapper.getDistNet(img_w, disp_max, hpatch, _BASE_FNET_:clone():double())
distNet:cuda()

err = {}
nbatch = 1;
for k, input, target  in supSet:subiter(prm['test_batch_size'], supSet:size()) do
  
  -- put validation set on gpu if needed  
  target = target:cuda()
  input[1] = input[1]:cuda();
  input[2] = input[2]:cuda();

  -- |test|
  err[nbatch], errCases = testFun.getTestAcc(distNet, input, target, prm['test_err_th'])
  gtByMax_, gtNum_  = testFun.getGraph(distNet, input, target, prm['test_err_th'])
  
  if( nbatch == 1 ) then
    gtByMax = gtByMax_
    gtNum  = gtNum_  
  else
    gtByMax = gtByMax_ + gtByMax
    gtNum  = gtNum_  + gtNum 
  end

  nbatch = nbatch + 1;

end

errPlot = torch.cumsum(gtByMax) * 100 / gtNum;
err = torch.cat(err,1)
errLt3 = torch.sum(err:lt(3)) * 100 / err:numel()

--test_acc_lt3  = torch.Tensor(test_acc_lt3)
--test_acc_lt3 = test_acc_lt3:mean()

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
    
    input, target  = supSet:index(torch.Tensor({lines[nline]}))
    local distMat, gtDistMat = testFun.getDist(distNet, 
                              {input[1]:cuda(), 
                               input[2]:cuda()}, 
                               target:cuda(), prm['test_err_th'])

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
logger:add{errLt3}


-- save error plot
local out = assert(io.open('work/' .. prm['test_result_fname'] .. '/'.. prm['test_result_fname'] .. '_plotErr', "w")) -- open a file for serialization
for i=1,errPlot:size(1) do
  out:write(errPlot[i])
  if i ~= errPlot:size(1) then
    out:write(", ")
  end
end
out:close()




