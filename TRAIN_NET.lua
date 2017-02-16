#!/usr/bin/env luajit

--[[ 

This is universal script for semi-supervised training of network 

It supports following networks:

mil
contrastive
contrastive-dprog

And following datasets:

mb
kitti
kitti_ext
kitti2015
kitti2015_ext

Typicall command line parameters
debug contrastive-dp acrt-kitti kitti

]]--

------------------------ read modules

-- standard modules
require 'gnuplot'
require 'optim'
require 'nn'
require 'image'
require 'torch'
require 'lfs'

-- cuda 
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true
   
-- custom
dofile('CHeadNetMulti.lua')  
trainerNet = dofile('CTrainerNet.lua')    
--
dofile('CContrastiveDP.lua');             
require 'libdprog'  -- C++ DP
--
dofile('CContrastive.lua');               
dofile('CMil.lua');               
dofile('CMilContrastive.lua');      

dofile('CUnsupMB.lua')                     
dofile('CUnsupKITTI.lua')

cnnMetric = dofile('CCNNMetric.lua');      
testFun = dofile('CTestUtils.lua');         -- Function that performs test on validation set
utils = dofile('utils.lua');              -- Utils for loading and visualization

----------------------------- parse input parameters

dbg = table.remove(arg, 1) 
method = table.remove(arg, 1)
arch = table.remove(arg, 1)
set = table.remove(arg, 1)

cmd = torch.CmdLine()

-- debug setting
dbg = dbg or 'debug';
method = method or 'mil'
arch = arch or 'fst-xxl'
set = set or 'kitti'

assert(method == 'mil' or method == 'contrastive' or method == 'mil-contrastive' or method == 'contrastive-dp')
assert(arch == 'fst-mb' or arch == 'fst-kitti' or arch == 'acrt-mb' or arch == 'acrt-kitti' or arch == 'fst-xxl')
assert(set == 'mb' or set == 'kitti' or set == 'kitti2015' or set == 'kitti2015_ext' or set == 'kitti_ext')

if dbg == 'normal' then
  -- for real training 
  cmd:option('-train_batch_size', 370)   
  cmd:option('-train_nb_batch', 100)        
  cmd:option('-train_nb_epoch', 1000)        
elseif dbg == 'tune' then
  cmd:option('-train_batch_size', 370)     
  cmd:option('-train_nb_batch', 100)        
  cmd:option('-train_nb_epoch', 5)        
else 
  cmd:option('-train_batch_size', 128)     
  cmd:option('-train_nb_batch', 1)        
  cmd:option('-train_nb_epoch', 10)        
end

-- semi-supervised method parameters
cmd:option('-loss_margin', 0.2)
cmd:option('-th_sup', 2) 
cmd:option('-th_occ', 1)    

-- dbg
if dbg == 'debug' then
  local timestamp = os.date("test-%Y_%m_%d_%X")
  cmd:option('-debug_fname', timestamp)
else
  cmd:option('-debug_fname', 'experiments')
end

cmd:option('-debug_start_from_net', '')

opt = cmd:parse(arg)
paths.mkdir('work/'..opt['debug_fname']); -- make output folder

-- log
local timestampBeg = os.date("%Y_%m_%d_%X")
cmd:log('work/' .. opt['debug_fname'] .. '/log_' .. opt['debug_fname'] .. '_' .. timestampBeg .. '.txt', opt, 'ascii')
print('Semi-suprevised training. Method: ' .. method .. ', arch: ' .. arch.. ', set: ' .. set.. ', mode: ' .. dbg)

math.randomseed(0); 
torch.manualSeed(0)

---------------------------- initialize / load networks 

local  hpatch
local _EMBED_NET_;
local _HEAD_NET_;

_EMBED_NET_ = cnnMetric.getEmbeddNet(arch)
_HEAD_NET_  = cnnMetric.getHeadNet(arch)
hpatch = cnnMetric.getHPatch(_EMBED_NET_)

_OPTIM_STATE_ = {}
_TRAIN_LOG_ = {}

-- try to read network
if( opt.debug_start_from_net ~= '' ) then
  if utils.file_exists(opt.debug_start_from_net) then
    local tmp = torch.load(opt.debug_start_from_net, 'ascii')      
    _EMBED_NET_ = tmp[1]
    _HEAD_NET_  = tmp[2]
    _OPTIM_STATE_ = tmp[3]
    _TRAIN_LOG_  = tmp[4]
    print('Continue training from network specified by user\n')
  else
    error('Could not find network specified by user\n')
  end
else
  local default_net = 'work/' .. opt['debug_fname'] .. '/metricNet_' .. opt['debug_fname'] .. '.t7'
  if utils.file_exists(default_net) then
    local tmp = torch.load(default_net, 'ascii')      
    _EMBED_NET_ = tmp[1]
    _HEAD_NET_  = tmp[2]
    _OPTIM_STATE_ = tmp[3]
    _TRAIN_LOG_ = tmp[4]
    print('Continue training from default network\n')
  else
    print('Could not find default network. Starting from the beggining.\n')
    
  end
end

-- put optimization state on gpu
if _OPTIM_STATE_.m then
  _OPTIM_STATE_.m = _OPTIM_STATE_.m:cuda()
  _OPTIM_STATE_.v = _OPTIM_STATE_.v:cuda()
  _OPTIM_STATE_.denom = _OPTIM_STATE_.denom:cuda()
end

_HEAD_NET_:cuda()
_EMBED_NET_:cuda();
cudnn.convert(_HEAD_NET_, cudnn)
cudnn.convert(_EMBED_NET_, cudnn)

_EMBED_PARAM_, _EMBED_GRAD_ = _EMBED_NET_:getParameters() 
_HEAD_PARAM_, _HEAD_GRAD_   = _HEAD_NET_:getParameters() 

---------------------------- initialize dataset

if set == 'kitti_ext' or set == 'kitti' then

  unsupSet = unsupKITTI('data/kitti_ext', set, hpatch);
  
elseif set == 'kitti2015' or set == 'kitti2015_ext' then
  
  unsupSet = unsupKITTI('data/kitti15_ext', set, hpatch);

elseif set == 'mb' then
  
  local metadata_fname = 'data/mb/meta.bin'
  local metadata = utils.fromfile(metadata_fname)
  local img_tab = {}
  local disp_tab = {}
  for n = 1,metadata:size(1) do
    local img_light_tab = {}
    light = 1
    while true do
      fname = ('data/mb/x_%d_%d.bin'):format(n, light)
      if not paths.filep(fname) then
        break
      end
      table.insert(img_light_tab, utils.fromfile(fname))
      light = light + 1
    end
    table.insert(img_tab, img_light_tab)
    fname = ('data/mb/dispnoc%d.bin'):format(n)
    if paths.filep(fname) then
      table.insert(disp_tab, utils.fromfile(fname))
    end
    if metadata[{n,3}] == -1 then -- fill max_disp for train set
      metadata[{n,3}] = disp_tab[n]:max()
    end
  end
    
  unsupSet = unsupMB(img_tab, metadata, hpatch)
  
end


-- |define optimization function|
feval = function(param)
  
  -- set network parameters
  if arch == 'fst-kitti' or arch == 'fst-mb' or arch == 'fst-xxl' then
    _EMBED_PARAM_:copy(param)
  else
    _EMBED_PARAM_:copy(param[{{1, _EMBED_PARAM_:size(1)}}])
    _HEAD_PARAM_:copy(param[{{1+_EMBED_PARAM_:size(1), param:size(1)}}])
  end
  
  local batch_size = #_TR_INPUT_[1];
  local epiRef, epiPos, epiNeg = unpack(_TR_INPUT_) -- epiNeg does not exist for  contrastive-max and contrastive-dprog
  
  _TR_LOSS_ = 0;   
  
  -- clear gradients
  _EMBED_GRAD_:zero()
  _HEAD_GRAD_:zero()
  
  local nb_comp_ttl = 0; 
  local criterion, tr_net
  
  for nsample = 1, batch_size do
  
    local sample_input = {}
    sample_input[1] = epiRef[nsample]
    sample_input[2] = epiPos[nsample]
    if epiNeg ~= nil then
      sample_input[3] = epiNeg[nsample]
    end
    
    if( _WIDTH_TAB_[nsample] ~= _WIDTH_TAB_[nsample-1] or _DISP_MAX_TAB_[nsample] ~= _DISP_MAX_TAB_[nsample-1] )  then
    
     
      if method == 'mil' then
      
        tr_net, criterion = trainerNet.getMil(_DISP_MAX_TAB_[nsample], _WIDTH_TAB_[nsample], opt['loss_margin'], _EMBED_NET_, _HEAD_NET_)  

      elseif method == 'contrastive' then
      
        tr_net, criterion =  trainerNet.getContrastive(_DISP_MAX_TAB_[nsample], _WIDTH_TAB_[nsample], opt['th_occ'], opt['loss_margin'], _EMBED_NET_, _HEAD_NET_) 
        
      elseif method == 'mil-contrastive' then
    
        tr_net, criterion =  trainerNet.getMilContrastive(_DISP_MAX_TAB_[nsample], _WIDTH_TAB_[nsample], opt['th_sup'],  opt['th_occ'], opt['loss_margin'], _EMBED_NET_, _HEAD_NET_)
        
      elseif( method == 'contrastive-dp' ) then
        
        tr_net, criterion =  trainerNet.getContrastiveDP(_DISP_MAX_TAB_[nsample], _WIDTH_TAB_[nsample], opt['th_sup'],  opt['th_occ'],  opt['loss_margin'], _EMBED_NET_, _HEAD_NET_)  
      
      end
    
      -- put training network on cuda
      tr_net:cuda()
      criterion:cuda()
      cudnn.convert(tr_net, cudnn)
       
    end
    
    tr_net:forward(sample_input)
    
    -- number of nonempty output tabels
    local nb_tables = #tr_net.output
    
    -- if number of nonempty output tables is 0, we can not do anything
    if nb_tables  > 0 then
    if tr_net.output[1][1]:numel() > 1  then
     
      -- make target array for every table, and simultaneously compute total number of samples
      local sample_target = {};
      for ntable = 1,nb_tables do
        local nb_comp = tr_net.output[ntable][1]:numel()
        nb_comp_ttl = nb_comp_ttl + nb_comp;
        sample_target[ntable] = nn.utils.addSingletonDimension(tr_net.output[ntable][1]:clone():fill(1),1) 
        sample_target[ntable] = sample_target[ntable]:cuda()
      end
      
      -- pass through criterion
      _TR_LOSS_ = _TR_LOSS_ + criterion:forward(tr_net.output, sample_target)

      -- backword pass
      tr_net:backward(sample_input, criterion:backward(tr_net.output, sample_target))
      collectgarbage()
    
    end
    end
     
  end
 
  _TR_LOSS_ = _TR_LOSS_ / nb_comp_ttl
  
  _EMBED_GRAD_:div(nb_comp_ttl);
  
  if arch == 'fst-kitti' or arch == 'fst-mb' or arch == 'fst-xxl'  then
    grad = _EMBED_GRAD_:clone()
  else 
    _HEAD_GRAD_:div(nb_comp_ttl);
    grad = torch.cat(_EMBED_GRAD_, _HEAD_GRAD_, 1)
  end
  
  return _TR_LOSS_, grad
end

-- initialize the parameters
if arch == 'fst-kitti' or arch == 'fst-mb' or arch == 'fst-xxl' then
  cur_param = _EMBED_PARAM_:clone();
else
  cur_param = torch.cat(_EMBED_PARAM_, _HEAD_PARAM_, 1)
end


local train_loss = 1/0;      
local time_per_batch = 1/0;   

for nepoch = 1, opt['train_nb_epoch'] do

  local start_time = os.time()
  local timestamp = os.date("%Y_%m_%d_%X_")
  
  -- compute validation error
  local valid_err 
  do 
 
    local net_fname = 'tmp/' ..timestamp.. '.t7';
 
    -- save current network 
    local network = {};
    network[1] = _EMBED_NET_
    network[2] = _HEAD_NET_
    torch.save(net_fname, network, 'ascii');
  
    local set_name
    if( set == 'kitti2015_ext' ) then
      set_name = 'kitti2015'
    elseif ( set == 'kitti_ext' ) then
      set_name = 'kitti';
    else
      set_name = set
    end
    
    local exec_str 
    if arch == 'fst-mb' or arch == 'fst-kitti' or arch == 'fst-xxl' then 
      exec_str = './main.lua ' .. set_name .. ' our -a test_te -sm_terminate cnn -net_fname ../mil-mc-cnn/' .. net_fname 
    else
      -- since accurate architecture is very slow, we compute validation error only using several images
      exec_str = './main.lua ' .. set_name .. ' our -a test_te -small_test 1 -sm_terminate cnn -net_fname ../mil-mc-cnn/' .. net_fname 
    end
 
    lfs.chdir('../mc-cnn')      -- switch current directory
    local handle = io.popen(exec_str)
    local result = handle:read("*a")
    local valid_err_str = string.gsub(result,'\n','');
    valid_err = tonumber(valid_err_str);
    lfs.chdir('../mil-mc-cnn')  
      
  end
    
  -- push new record to log
  local trainingLog = {}
  trainingLog['train_loss'] = train_loss
  trainingLog['valid_err']  = valid_err
  trainingLog['dt']         = time_per_batch
  table.insert(_TRAIN_LOG_, trainingLog)
  
  -- save network 
  do
    local network = {}
    network[1] = _EMBED_NET_
    network[2] = _HEAD_NET_
    local optim_state = {}
    if nepoch > 1 then
      optim_state.t = _OPTIM_STATE_.t;
      optim_state.m = _OPTIM_STATE_.m:double();
      optim_state.v = _OPTIM_STATE_.v:double();
      optim_state.denom = _OPTIM_STATE_.denom:double();  
    end
    network[3] = optim_state
    network[4] = _TRAIN_LOG_
    
    local net_fname = 'work/' .. opt['debug_fname'] .. '/metricNet_' .. opt['debug_fname'] .. timestamp.. '.t7'; -- history log 
    torch.save(net_fname, tmp, 'ascii');
    
    local net_fname = 'work/' .. opt['debug_fname'] .. '/metricNet_' .. opt['debug_fname'] .. '.t7';             -- current state log
    torch.save(net_fname, tmp, 'ascii');
  end
    
  -- save log
  local time
  do
    local log_fname = 'work/' .. opt['debug_fname'] .. '/err_' .. opt['debug_fname'] ..'_'.. timestampBeg.. '.txt';
    local f = io.open(log_fname, 'w')
    f:write(string.format("%d, %f, %f, %f\n", 1, 0, 1/0, _TRAIN_LOG_[1].valid_err))
    time = 0
    for i = 2, #_TRAIN_LOG_ do
      time  = time  + _TRAIN_LOG_[i].dt
      f:write(string.format("%d, %f, %f, %f\n", i, time, _TRAIN_LOG_[i].train_loss, _TRAIN_LOG_[i].valid_err))
    end
    f:close()
  end
  
  -- print log
  print(string.format("epoch %d, time = %f, train_loss = %f, valid_err = %f\n", nepoch, time, _TRAIN_LOG_[#_TRAIN_LOG_].train_loss, _TRAIN_LOG_[#_TRAIN_LOG_].valid_err))
    
  -- go through batches
  batch_loss = {}
  for nbatch = 1, opt['train_nb_batch'] do
    
   -- get batch
   _TR_INPUT_, _WIDTH_TAB_, _DISP_MAX_TAB_ = unsupSet:get(opt['train_batch_size'])   
           
   -- some network we need only two epopolar lines
   if method == 'contrastive' or method == 'contrastive-dp' then
      _TR_INPUT_[3] = nil
   end
      
   -- put on gpu
   for nsample = 1, opt.train_batch_size do
    for i = 1,#_TR_INPUT_ do
      _TR_INPUT_[i][nsample] = _TR_INPUT_[i][nsample]:cuda()
    end
   end
       
   -- optimize 
   optim.adam(feval, cur_param, {}, _OPTIM_STATE_)    
   
   -- update parameters
  if arch == 'fst-kitti' or arch == 'fst-mb' or arch == 'fst-xxl' then
    _EMBED_PARAM_:copy(cur_param)
  else
    _EMBED_PARAM_:copy(cur_param[{{1, _EMBED_PARAM_:size(1)}}])
    _HEAD_PARAM_:copy(cur_param[{{1+_EMBED_PARAM_:size(1), cur_param:size(1)}}])
  end
  
   table.insert(batch_loss, _TR_LOSS_)
   
   collectgarbage()
   
  end
    
  -- compute time
  local end_time = os.time()
  time_per_batch = os.difftime(end_time, start_time);
  
  -- compute trainin loss for epoch
  train_loss = torch.Tensor(batch_loss):mean();
  
end






