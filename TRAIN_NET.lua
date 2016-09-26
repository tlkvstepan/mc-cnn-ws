--[[ 
This is universal script for semi-supervised training of network 

It supports following networks:

-mil-max
-mil-dprog
-contrast-max
-contrast-dprog
-mil-contrast-max
-mil-contrast-dprog

]]--

require 'torch'

-- |read input parameters|
-- fist argument is training net architecture
arch = table.remove(arg, 1) 
cmd = torch.CmdLine()

assert(arch == 'mil-max' or arch == 'mil-dprog' or arch == 'contrast-max' or arch == 'contrast-dprog' or arch == 'mil-contrast-max' or arch =='mil-contrast-dprog')

-- optimization parameters parameters
cmd:option('-valid_set_size', 350)       
cmd:option('-train_batch_size', 1024)     
cmd:option('-train_epoch_size', 1024*100) 
cmd:option('-train_nb_epoch', 300)        

-- training network parameters
cmd:option('-loss_margin', 0.2)
cmd:option('-dist_min', 2)

-- feature network parameters
cmd:option('-net_nb_feature', 64)
cmd:option('-net_kernel', 3)
cmd:option('-net_nb_layers', 4)

-- debug
cmd:option('-debug_err_th', 3)
cmd:option('-debug_fname', 'test')
cmd:option('-debug_gpu_on', true)
cmd:option('-debug_save_on', true)
cmd:option('-debug_start_from_timestamp', '')

prm = cmd:parse(arg)
prm['arch'] = arch

paths.mkdir('work/'..prm['debug_fname']); -- make output folder

print('Semi-suprevised training with ' .. arch .. ' arhitecture \n')
  
-- |load modules|

-- standard modules
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

print('Parameters of the procedure : \n')
utils.printTable(prm)

if( prm['debug_gpu_on'] ) then            
  require 'cunn'
end

-- Set random seeds for math and torch for repeatability
math.randomseed(0); 
torch.manualSeed(0)

-- |read trainng data| (KITTI)
local img1_arr = torch.squeeze(utils.fromfile('data/KITTI12/x0.bin'));
local img2_arr = torch.squeeze(utils.fromfile('data/KITTI12/x1.bin'));
local disp_arr = torch.round(torch.squeeze(utils.fromfile('data/KITTI12/dispnoc.bin')));

local disp_max = disp_arr:max()
local img_w = img1_arr:size(3);

-- |define test and training networks|
-- If we choose to start from timestamp, when try to read pre-trained base feature net
local fnet_fname = 'work/' .. prm['debug_fname'] .. '/fnet_' .. prm['debug_start_from_timestamp'] .. '_' .. prm['debug_fname'] .. '.t7'
local optim_fname = 'work/' .. prm['debug_fname'] .. '/optim_' .. prm['debug_start_from_timestamp'] .. '_'.. prm['debug_fname'] .. '.t7'
if utils.file_exists(fnet_fname) and utils.file_exists(optim_fname) then
  print('Continue training. Please delete the network file if you wish to start from the beggining\n')
  _BASE_FNET_= torch.load(fnet_fname, 'ascii')
  hpatch = ( utils.get_window_size(_BASE_FNET_)-1 )/ 2
  _OPTIM_STATE_ =  torch.load(optim_fname, 'ascii')
else
  print('Start training from the begining\n')
  _BASE_FNET_, hpatch = baseNet.get(prm['net_nb_layers'], prm['net_nb_feature'], prm['net_kernel'])
  _OPTIM_STATE_ = {}
end

-- make training network (note that parameters are copied from base feature network)
if arch == 'mil-max' then
  _TR_NET_, _CRITERION_ =  netWrapper.getMilMax(img_w, disp_max, hpatch, prm['loss_margin'], _BASE_FNET_)
elseif arch == 'mil-dprog' then
   _TR_NET_, _CRITERION_ =  netWrapper.getMilDprog(img_w, disp_max, hpatch, prm['loss_margin'], _BASE_FNET_)
elseif arch == 'contrast-max' then
  _TR_NET_, _CRITERION_ = netWrapper.getContrastMax(img_w, disp_max, hpatch, prm['dist_min'], prm['loss_margin'], _BASE_FNET_)  
elseif arch == 'contrast-dprog' then
  _TR_NET_, _CRITERION_ = netWrapper.getContrastDprog(img_w, disp_max, hpatch, prm['dist_min'], prm['loss_margin'], _BASE_FNET_)
elseif arch == 'mil-contrast-max' then
  _TR_NET_, _CRITERION_ = netWrapper.getMilContrastMax(img_w, disp_max, hpatch, prm['dist_min'], prm['loss_margin'], _BASE_FNET_)
elseif arch == 'mil-contrast-dprog' then
  _TR_NET_, _CRITERION_ = netWrapper.getMilContrastDprog(img_w, disp_max, hpatch, prm['dist_min'], prm['loss_margin'], _BASE_FNET_)
end

-- put training net, base net, criterion and state of optimizer on gpu if needed
if prm['debug_gpu_on'] then
  _TR_NET_:cuda()
  _BASE_FNET_:cuda()
  _CRITERION_:cuda()
  if _OPTIM_STATE_.m then
    _OPTIM_STATE_.m = _OPTIM_STATE_.m:cuda()
    _OPTIM_STATE_.v = _OPTIM_STATE_.v:cuda()
    _OPTIM_STATE_.denom = _OPTIM_STATE_.denom:cuda()
  end
end

-- get training and base network parametesr
_BASE_PPARAM_ = _BASE_FNET_:getParameters() 
_TR_PPARAM_, _TR_PGRAD_ = _TR_NET_:getParameters()

-- |define datasets|

local unsupSet = unsup3EpiSet(img1_arr, img2_arr, hpatch, disp_max);
local supSet = sup2EpiSet(img1_arr[{{1,194},{},{}}], img2_arr[{{1,194},{},{}}], disp_arr[{{1,194},{},{}}], hpatch);
supSet:shuffle()  -- shuffle to have patches from all images

-- get validation set from supervised set
_VA_INPUT_, _VA_TARGET_ = supSet:index(torch.range(1, prm['valid_set_size']))

-- put validation set on gpu if needed  
if prm['debug_gpu_on'] then
  _VA_TARGET_ = _VA_TARGET_:cuda()
  _VA_INPUT_[1] = _VA_INPUT_[1]:cuda();
  _VA_INPUT_[2] = _VA_INPUT_[2]:cuda();
end

-- |define optimization function|
feval = function(x)

  -- set net parameters
  _TR_PPARAM_:copy(x)

  -- clear gradients
  _TR_PGRAD_:zero()

  local batch_size = _TR_INPUT_[1]:size(1);
  local epiRef, epiPos, epiNeg = unpack(_TR_INPUT_) -- epiNeg does not exist for  contrast-max and contrast-dprog

  _TR_ERR_ = 0;   
  for nsample = 1, batch_size do
    
    local sample_input = {}
    sample_input[1] = epiRef[{{nsample},{},{}}]
    sample_input[2] = epiPos[{{nsample},{},{}}]
    if epiNeg ~= nil then
      sample_input[3] = epiNeg[{{nsample},{},{}}]
    end
  
    local sample_target = _TR_TARGET_[{{nsample},{}}]    
    
    -- forward pass
    _TR_NET_:forward(sample_input)
    _TR_ERR_ = _TR_ERR_ + _CRITERION_:forward(_TR_NET_.output, sample_target)

    -- backword pass
    _TR_NET_:backward(sample_input, _CRITERION_:backward(_TR_NET_.output, sample_target))
     collectgarbage()
     
  end
  _TR_ERR_ = _TR_ERR_ / prm['train_batch_size']
  _TR_PGRAD_:div(prm['train_batch_size']);

  return _TR_ERR_, _TR_PGRAD_      
end

-- |save debug info|
if prm['debug_save_on'] then
  
  -- save train parameters
  local timestamp = os.date("%Y_%m_%d_%X_")
  torch.save('work/' .. prm['debug_fname'] .. '/params_' .. timestamp .. prm['debug_fname'] .. '.t7', prm, 'ascii');
  
end
    
-- |define logger|
if prm['debug_save_on'] then
  logger = optim.Logger('work/' .. prm['debug_fname'] .. '/'.. prm['debug_fname'], true)
  logger:setNames{'Training loss', 
    'Accuracy (<3 disparity err)'}
  logger:style{'+-',
    '+-',
    '+-'}
end    

-- |optimize network|   
local start_time = os.time()

for nepoch = 1, prm['train_nb_epoch'] do

  nsample = 0;
  sample_err = {}
  local train_err = 0

  for k, input  in unsupSet:sampleiter(prm['train_batch_size'], prm['train_epoch_size']) do

   _TR_INPUT_ = {}
   _TR_INPUT_[1] = input[1]
   _TR_INPUT_[2] = input[2]
   if arch == 'mil-contrast-max' or arch == 'mil-max' or arch == 'mil-dprog' or arch == 'mil-contrast-dprog' then
    _TR_INPUT_[3] = input[3]
   end
   _TR_TARGET_ =  torch.ones(prm['train_batch_size'], (img_w - disp_max - 2*hpatch));  

   -- if gpu avaliable put batch on gpu
   if prm['debug_gpu_on'] then
     for i = 1,#_TR_INPUT_ do
     _TR_INPUT_[i] = _TR_INPUT_[i]:cuda()
     end
     _TR_TARGET_ = _TR_TARGET_:cuda()
   end
    
   optim.adam(feval, _TR_PPARAM_, {}, _OPTIM_STATE_)    
   table.insert(sample_err, _TR_ERR_)

end

train_err = torch.Tensor(sample_err):mean();

-- validation
_BASE_PPARAM_:copy(_TR_PPARAM_)
local distNet = netWrapper.getDistNet(img_w, disp_max, hpatch, _BASE_FNET_:clone():double())
if prm['debug_gpu_on'] then
  distNet:cuda()
end
local test_acc_lt3, errCases = testFun.getTestAcc(distNet, _VA_INPUT_, _VA_TARGET_, prm['debug_err_th'])

local end_time = os.time()
local time_diff = os.difftime(end_time,start_time);

-- save debug info
if prm['debug_save_on'] then
  
  local timestamp = os.date("%Y_%m_%d_%X_")

  -- save errorneous test samples
  local fail_img = utils.vis_errors(errCases[1], errCases[2], errCases[3], errCases[4])
  image.save('work/' .. prm['debug_fname'] .. '/error_cases_' .. timestamp .. prm['debug_fname'] .. '.png',fail_img)

  -- save net (we save all as double tensor)
  torch.save('work/' .. prm['debug_fname'] .. '/fnet_' .. timestamp .. prm['debug_fname'] .. '.t7', _BASE_FNET_:clone():double(), 'ascii');

  -- save optim state (we save all as double tensor)
  local optim_state_host ={}
  optim_state_host.t = _OPTIM_STATE_.t;
  optim_state_host.m = _OPTIM_STATE_.m:clone():double();
  optim_state_host.v = _OPTIM_STATE_.v:clone():double();
  optim_state_host.denom = _OPTIM_STATE_.denom:clone():double();
  torch.save('work/' .. prm['debug_fname'] .. '/optim_'.. timestamp .. prm['debug_fname'] .. '.t7', optim_state_host, 'ascii');

  -- save log
  logger:add{train_err, test_acc_lt3}
  logger:plot()

  -- save distance matrices
  local lines = {3,9}
  for nline = 1,#lines do
    
    local distMat, gtDistMat = testFun.getDist(distNet, {_VA_INPUT_[1][{{lines[nline]},{},{}}], _VA_INPUT_[2][{{lines[nline]},{},{}}]}, _VA_TARGET_[{{lines[nline]},{},{}}], prm['debug_err_th'])

    local gtDistMat = 1-utils.scale2_01(gtDistMat)

    local distMat = utils.softmax(distMat:squeeze())
    local distMat = 1-utils.scale2_01(distMat)

    local r = distMat:clone()
    local g = distMat:clone()
    local b = distMat:clone()

    r[gtDistMat:eq(0)] = 0 
    g[gtDistMat:eq(0)] = 1
    b[gtDistMat:eq(0)] = 0

    im = torch.cat({nn.utils.addSingletonDimension(r,1), nn.utils.addSingletonDimension(g,1), nn.utils.addSingletonDimension(b,1)}, 1)
   
   image.save('work/' .. prm['debug_fname'] .. '/dist_' ..  string.format("line%i_",lines[nline])  .. timestamp .. prm['debug_fname'] .. '.png',im)
  end
end

print(string.format("epoch %d, time = %f, train_err = %f, test_acc = %f", nepoch, time_diff, train_err, test_acc_lt3))
collectgarbage()

end





