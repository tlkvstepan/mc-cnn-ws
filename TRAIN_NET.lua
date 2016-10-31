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
-- fist   - argument is debug mode
-- second - training wrapper architecture
-- third  - training set
dbg = table.remove(arg, 1) 
if dbg == 'debug' then
  arch = table.remove(arg, 1)
  dbg = true
else
  arch = dbg
  dbg = false
end
set = table.remove(arg, 1)

cmd = torch.CmdLine()

assert(arch == 'mil-max' or arch == 'mil-dprog' or arch == 'contrast-max' or arch == 'contrast-dprog' or arch == 'mil-contrast-max' or arch =='mil-contrast-dprog')

-- optimization parameters parameters
if not dbg then
  cmd:option('-valid_set_size', 100)        -- 100 epi lines      
  cmd:option('-train_batch_size', 360)      -- 342 one image in KITTI
  cmd:option('-train_nb_batch', 400)        -- 400 all images in KITTI
  cmd:option('-train_nb_epoch', 100)        -- 35 times all images in KITTI
else
  cmd:option('-valid_set_size', 100)       -- 100 epi lines      
  cmd:option('-train_batch_size', 128)     -- 129 one image in KITTI
  cmd:option('-train_nb_batch', 50)       -- 50
  cmd:option('-train_nb_epoch', 10)        -- 10
end
cmd:option('-train_set_prc', 0.1)        -- 100 epi lines      
 
-- training network parameters
cmd:option('-loss_margin', 0.2)
cmd:option('-maxsup_th', 2) -- 2
cmd:option('-occ_th', 1)    -- 1

-- feature network parameters
-- (we have different default networks for mb and kitti)
cmd:option('-net_nb_feature', 64)
cmd:option('-net_kernel', 3)
if( set == 'mb' ) then
  cmd:option('-net_nb_layers', 5)
else
  cmd:option('-net_nb_layers', 4)
end

-- debug
cmd:option('-debug_err_th', 3)
cmd:option('-debug_fname', 'test')
cmd:option('-debug_gpu_on', 1)
cmd:option('-debug_start_from_fnet', '')
cmd:option('-debug_start_from_optim', '')

prm = cmd:parse(arg)
prm['arch'] = arch
if prm['debug_gpu_on'] >= 1 then
   prm['debug_gpu_on'] = true
else
   prm['debug_gpu_on'] = false
end

paths.mkdir('work/'..prm['debug_fname']); -- make output folder
print('Semi-suprevised training ' .. arch .. ' arhitecture and ' .. set .. ' set\n')
  
-- |load modules|

-- standard modules
require 'gnuplot'
require 'optim'
require 'nn'
require 'image'

-- Custom modules
dofile('CAddMatrix.lua')                  -- Module that adds constant matrix to the input (I use it for masking purposes)

require 'libdprog'                        -- C++ module for dynamic programming
dofile('CContrastDprog.lua');             -- Contrastive dynamic programming module
dofile('CContrastMax.lua');               -- Contrastive max-2ndMax module
dofile('CMilDprog.lua');
dofile('CMilContrastDprog.lua')
dofile('DataLoader.lua');                 -- Parent class for dataloaders

-- unsupervised set difinition
dofile('CUnsupSet.lua')     
dofile('CUnsupMB.lua')    -- MB

dofile('CUnsupKITTI_HD.lua')

--dofile('CUnsup3EpiSet.lua');              -- Unsupervised training set loader
dofile('CSup2EpiSet.lua');          -- Supervised validation set loader

baseNet = dofile('CBaseNet.lua');         -- Function that makes base net
netWrapper = dofile('CNetWrapper.lua')    -- Function that "wrap" base net into training net
testFun = dofile('CTestUtils.lua');         -- Function that performs test on validation set

utils = dofile('utils.lua');              -- Utils for loading and visualization

print('Parameters of the procedure : \n')
utils.printTable(prm)

if( prm['debug_gpu_on'] ) then            
  require 'cunn'
  require 'cudnn'
end

math.randomseed(0); 
torch.manualSeed(0)

-- |define feature network|
-- If we choose to start from timestamp, when try to read pre-trained base feature net
local fnet_fname = prm['debug_start_from_fnet'] 
local optim_fname = prm['debug_start_from_optim'] 
_BASE_FNET_, hpatch = baseNet.get(prm['net_nb_layers'], prm['net_nb_feature'], prm['net_kernel'])
_OPTIM_STATE_ = {}
if utils.file_exists(fnet_fname) and utils.file_exists(optim_fname)   then
  print('Continue training.\n')
  _BASE_FNET_= torch.load(fnet_fname, 'ascii')
  hpatch = ( utils.get_window_size(_BASE_FNET_)-1 )/ 2
  _OPTIM_STATE_ = torch.load(optim_fname, 'ascii')
else
  print('Start training from the begining\n')
end

-- put base net and state of optimizer on gpu if needed
if prm['debug_gpu_on'] then
  if _OPTIM_STATE_.m then
    _OPTIM_STATE_.m = _OPTIM_STATE_.m:cuda()
    _OPTIM_STATE_.v = _OPTIM_STATE_.v:cuda()
    _OPTIM_STATE_.denom = _OPTIM_STATE_.denom:cuda()
  end
end

-- get training and base network parametesr
_BASE_PPARAM_ = _BASE_FNET_:getParameters() 

-- |read data and set up training and validation sets|
if set == 'kitti_ext' or set == 'kitti'  then
  
  local x0_fname = 'data/kitti/x0.bin'
  local x1_fname = 'data/kitti/x1.bin'
  local dispnoc_fname  = 'data/kitti/dispnoc.bin'
  local nb_tr = 194
  local img1_arr = torch.squeeze(utils.fromfile(x0_fname));
  local img2_arr = torch.squeeze(utils.fromfile(x1_fname));
  local disp_arr = torch.round(torch.squeeze(utils.fromfile(dispnoc_fname)));
  disp_max = disp_arr:max()
  img_w = img1_arr:size(3);
  
  unsupSet = unsupKITTI_HD('data/kitti_ext', set, hpatch);
  
  supSet = sup2EpiSet(img1_arr[{{1,nb_tr},{},{}}], img2_arr[{{1,nb_tr},{},{}}], disp_arr[{{1,nb_tr},{},{}}], hpatch);
  supSet:shuffle()  -- shuffle to have patches from all images

elseif set == 'kitti15' or set == 'kitti15_ext' then
  
  local x0_fname = 'data/kitti15/x0.bin'
  local x1_fname = 'data/kitti15/x1.bin'
  local dispnoc_fname  = 'data/kitti15/dispnoc.bin'
  local nb_tr = 200
  local img1_arr = torch.squeeze(utils.fromfile(x0_fname));
  local img2_arr = torch.squeeze(utils.fromfile(x1_fname));
  local disp_arr = torch.round(torch.squeeze(utils.fromfile(dispnoc_fname)));
  disp_max = disp_arr:max()
  img_w = img1_arr:size(3);
  
  unsupSet = unsupKITTI_HD('data/kitti15_ext', set, hpatch);
  
  supSet = sup2EpiSet(img1_arr[{{1,nb_tr},{},{}}], img2_arr[{{1,nb_tr},{},{}}], disp_arr[{{1,nb_tr},{},{}}], hpatch);
  supSet:shuffle()  -- shuffle to have patches from all images

elseif set == 'mb' then
  
  local metadata_fname = 'data/MB/meta.bin'
  local metadata = utils.fromfile(metadata_fname)
  local img_tab = {}
  local disp_tab = {}
  for n = 1,metadata:size(1) do
    local img_light_tab = {}
    light = 1
    while true do
      fname = ('data/MB/x_%d_%d.bin'):format(n, light)
      if not paths.filep(fname) then
        break
      end
      table.insert(img_light_tab, utils.fromfile(fname))
      light = light + 1
    end
    table.insert(img_tab, img_light_tab)
    fname = ('data/MB/dispnoc%d.bin'):format(n)
    if paths.filep(fname) then
      table.insert(disp_tab, utils.fromfile(fname))
    end
  end
    
  unsupSet = unsupMB(img_tab, metadata, hpatch)
  
end
unsupSet:subset(prm['train_set_prc'])

---- |define datasets|
---- we want to have same training and test set all the time 

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
  
  local nb_comp_ttl = 0;
  for nsample = 1, batch_size do
    
    local sample_input = {}
    sample_input[1] = epiRef[{{nsample},{},{}}]
    sample_input[2] = epiPos[{{nsample},{},{}}]
    if epiNeg ~= nil then
      sample_input[3] = epiNeg[{{nsample},{},{}}]
    end
  
    -- forward pass through net
    _TR_NET_:forward(sample_input)
    
    -- find number of nonempty output tabels
    local nb_tables = #_TR_NET_.output
    
    -- if nuber of nonempty output tables is 0, we can not do anything
    if nb_tables ~= 0 then
    
      -- make target array for every table, and simultaneously compute 
      -- total number of samples
      local sample_target = {};
      for ntable = 1,nb_tables do
        local nb_comp = _TR_NET_.output[ntable][1]:numel()
        nb_comp_ttl = nb_comp_ttl + nb_comp;
        sample_target[ntable] = nn.utils.addSingletonDimension(_TR_NET_.output[ntable][1]:clone():fill(1),1) 
      end
      
      -- pass through criterion
      _TR_ERR_ = _TR_ERR_ + _CRITERION_:forward(_TR_NET_.output, sample_target)

      -- backword pass
      _TR_NET_:backward(sample_input, _CRITERION_:backward(_TR_NET_.output, sample_target))
       collectgarbage()
     
     end
     
  end
  
  _TR_ERR_ = _TR_ERR_ / nb_comp_ttl
  _TR_PGRAD_:div(nb_comp_ttl);

  return _TR_ERR_, _TR_PGRAD_      
end

-- |save debug info|
local timestamp = os.date("%Y_%m_%d_%X_")
torch.save('work/' .. prm['debug_fname'] .. '/params_' .. timestamp .. prm['debug_fname'] .. '.t7', prm, 'ascii');
    
-- |define logger|
logger = optim.Logger('work/' .. prm['debug_fname'] .. '/'.. prm['debug_fname'], true)
logger:setNames{'Training loss', 'Accuracy (<3 disparity err)'}
logger:style{'+-', '+-'}

-- |optimize network|   
local start_time = os.time()

-- go through epoches
for nepoch = 1, prm['train_nb_epoch'] do

  nsample = 0;
  sample_err = {}
  local train_err = 0
  
  -- go through batches
  for nbatch = 1, prm['train_nb_batch'] do
    
   -- get epipolar lines
   _TR_INPUT_, width, dispMax = unsupSet:get(prm['train_batch_size'])   
   
   -- some network need only two epopolar lines
   if arch == 'contrast-max' or arch == 'contrast-dprog' then
      _TR_INPUT_[3] = nil
   end
   
   -- make training network (note that parameters are copied from base feature network)
   if arch == 'mil-max' then
     _TR_NET_, _CRITERION_ =  netWrapper.getMilMax(width, dispMax, hpatch, prm['loss_margin'], _BASE_FNET_)
   elseif arch == 'mil-dprog' then
      _TR_NET_, _CRITERION_ =  netWrapper.getMilDprog(width, dispMax, hpatch, prm['occ_th'], prm['loss_margin'], _BASE_FNET_)
   elseif arch == 'contrast-max' then
     _TR_NET_, _CRITERION_ = netWrapper.getContrastMax(width, dispMax, hpatch, prm['maxsup_th'], prm['loss_margin'], _BASE_FNET_)  
   elseif arch == 'contrast-dprog' then
     _TR_NET_, _CRITERION_ = netWrapper.getContrastDprog(width, dispMax, hpatch, prm['maxsup_th'],  prm['occ_th'], prm['loss_margin'], _BASE_FNET_)
   elseif arch == 'mil-contrast-max' then
     _TR_NET_, _CRITERION_ = netWrapper.getMilContrastMax(width, dispMax, hpatch, prm['maxsup_th'], prm['loss_margin'], _BASE_FNET_)
   elseif arch == 'mil-contrast-dprog' then
     _TR_NET_, _CRITERION_ = netWrapper.getMilContrastDprog(width, dispMax, hpatch, prm['maxsup_th'],  prm['occ_th'], prm['loss_margin'], _BASE_FNET_)
   end

   -- put on gpu
   if prm['debug_gpu_on'] then
     _TR_NET_:cuda()
     cudnn.convert(_TR_NET_, cudnn)
     _CRITERION_:cuda()
     for i = 1,#_TR_INPUT_ do
     _TR_INPUT_[i] = _TR_INPUT_[i]:cuda()
     end
   end
   
   -- get training network
   _TR_PPARAM_, _TR_PGRAD_ = _TR_NET_:getParameters() 
    
   -- optimize 
   optim.adam(feval, _TR_PPARAM_, {}, _OPTIM_STATE_)    
   table.insert(sample_err, _TR_ERR_)

   -- save new parameteres 
   _BASE_PPARAM_:copy(_TR_PPARAM_)

end

train_err = torch.Tensor(sample_err):mean();

local distNet = netWrapper.getDistNet(img_w, disp_max, hpatch, _BASE_FNET_:clone():double())
if prm['debug_gpu_on'] then
  distNet:cuda()
  cudnn.convert(distNet, cudnn)
end

local dispErr, errCases = testFun.getTestAcc(distNet, _VA_INPUT_, _VA_TARGET_, prm['debug_err_th'])
test_acc_lt3 = dispErr[dispErr:lt(prm['debug_err_th'])]:numel() * 100 / dispErr:numel();


local end_time = os.time()
local time_diff = os.difftime(end_time,start_time);

-- save debug info

  
  local timestamp = os.date("%Y_%m_%d_%X_")

  -- save errorneous test samples
  local fail_img = utils.vis_errors(errCases[1], errCases[2], errCases[3], errCases[4])
  image.save('work/' .. prm['debug_fname'] .. '/error_cases_' .. timestamp .. prm['debug_fname'] .. '.png',fail_img)

  -- save net (we save all as double tensor)
  torch.save('work/' .. prm['debug_fname'] .. '/fnet_' .. timestamp .. prm['debug_fname'] .. '.t7', _BASE_FNET_:clone():double(), 'ascii');

  -- save optim state (we save all as double ten_TE_TARGET_sor)
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
  for nline = 1,2 do
    
    local distMat, gtDistMat = testFun.getDist(distNet, {_VA_INPUT_[1][{{lines[nline]},{},{}}], _VA_INPUT_[2][{{lines[nline]},{},{}}]}, _VA_TARGET_[{{lines[nline]},{},{}}], prm['debug_err_th'])

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
   
   image.save('work/' .. prm['debug_fname'] .. '/dist_' ..  string.format("line%i_",lines[nline])  .. timestamp .. prm['debug_fname'] .. '.png',im)
  end

print(string.format("epoch %d, time = %f, train_err = %f, test_acc = %f", nepoch, time_diff, train_err, test_acc_lt3))
collectgarbage()

end





