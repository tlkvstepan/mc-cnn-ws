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

]]--


-- --
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

-- custom
dofile('CAddMatrix.lua')                  -- Module that adds constant matrix to the input (I use it for masking purposes)
require 'libdprog'                        -- C++ module for dynamic programming
dofile('CContrastDprog.lua');             -- Contrastive dynamic programming module
dofile('CContrastMax.lua');               -- Contrastive max-2ndMax module
--dofile('CMilDprog.lua');
--dofile('CMilContrastDprog.lua')
--dofile('DataLoader.lua');                 -- Parent class for dataloaders
--dofile('CUnsupSet.lua')     
dofile('CUnsupMB.lua')                     -- MB
dofile('CUnsupKITTI.lua')
--dofile('CSup2EpiSet.lua');                 -- Supervised validation set loader
nnMetric = dofile('nnMetric.lua');         -- Function that makes base net
trainerNet = dofile('CTrainerNet.lua')    -- Function that "wrap" base net into training net
testFun = dofile('CTestUtils.lua');         -- Function that performs test on validation set
utils = dofile('utils.lua');              -- Utils for loading and visualization

----------------------------- parse input parameters
dbg = table.remove(arg, 1) 
method = table.remove(arg, 1)
metric = table.remove(arg, 1)
set = table.remove(arg, 1)

cmd = torch.CmdLine()

assert(method == 'mil' or method == 'contrastive' or method == 'contrastive-dp')
assert(metric == 'mc-cnn-fst-mb' or metric == 'mc-cnn-fst-kitti')
assert(set == 'mb' or set == 'kitti' or set == 'kitti2015' or set == 'kitti2015_ext' or set == 'kitti_ext')

if dbg == 'normal' then
  -- for real training 
  cmd:option('-train_batch_size', 370)   
  cmd:option('-train_nb_batch', 100)        
  cmd:option('-train_nb_epoch', 1000)        
elseif dbg == 'tune' then
  cmd:option('-train_batch_size', 370)     
  cmd:option('-train_nb_batch', 50)        
  cmd:option('-train_nb_epoch', 4)        
else 
  cmd:option('-train_batch_size', 32)     
  cmd:option('-train_nb_batch', 10)        
  cmd:option('-train_nb_epoch', 7)        
end

-- semi-supervised method parameters
cmd:option('-loss_margin', 0.2)
cmd:option('-th_sup', 2) 
cmd:option('-th_occ', 1)    

-- dbg
cmd:option('-debug_fname', 'test')
cmd:option('-debug_start_from_net', '')

opt = cmd:parse(arg)
paths.mkdir('work/'..opt['debug_fname']); -- make output folder

-- log
local timestampBeg = os.date("%Y_%m_%d_%X")
cmd:log('work/' .. opt['debug_fname'] .. '/log_' .. opt['debug_fname'] .. '_' .. timestampBeg .. '.txt', opt, 'ascii')
print('Semi-suprevised training. Method: ' .. method .. ', metric: ' .. metric.. ', set: ' .. set.. ', mode: ' .. dbg)

math.randomseed(0); 
torch.manualSeed(0)

---------------------------- initialize / load networks 

local _METRIC_NET_, hpatch
_METRIC_NET_, hpatch = nnMetric.get(metric)
_OPTIM_STATE_ = {}
_TRAIN_LOG_ = {}

-- try to read network
if( opt.debug_start_from_net ~= '' ) then
  if utils.file_exists(opt.debug_start_from_net) then
    local tmp = torch.load(opt.debug_start_from_net, 'ascii')      
    _METRIC_NET_ = tmp[1]
    _OPTIM_STATE = tmp[2]
    _TRAIN_LOG_  = tmp[3]
    print('Continue training from network specified by user\n')
  else
    error('Could not find network specified by user\n')
  end
else
  local default_net = 'work/' .. opt['debug_fname'] .. '/metricNet_' .. opt['debug_fname'] .. '.t7'
  if utils.file_exists(default_net) then
    local tmp = torch.load(default_net, 'ascii')      
    _METRIC_NET_ = tmp[1]
    _OPTIM_STATE = tmp[2]
    _TRAIN_LOG_  = tmp[3]
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

-- metric on cuda
_METRIC_NET_:cuda()

-- get metric network parameter
_METRIC_PPARAM_, _METRIC_PGRAD_ = _METRIC_NET_:getParameters() 

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
feval = function(x)

  local batch_size = #_TR_INPUT_[1];
  local epiRef, epiPos, epiNeg = unpack(_TR_INPUT_) -- epiNeg does not exist for  contrastive-max and contrastive-dprog
  
  _TR_LOSS_ = 0;   
  
  -- clear gradients
  _METRIC_PGRAD_:zero()
  
  local nb_comp_ttl = 0; 
  local criterion, tr_net
  
  for nsample = 1, batch_size do
  
    local sample_input = {}
    sample_input[1] = epiRef[nsample]
    sample_input[2] = epiPos[nsample]
    if epiNeg ~= nil then
      sample_input[3] = epiNeg[nsample]
    end
    
    if( _WIDTH_TAB_[nsample] ~= _WIDTH_TAB_[nsample-1] or _DISP_TAB_[nsample] ~= _DISP_TAB_[nsample-1] )  then
    
      if( method == 'contrastive-dp' ) then
        tr_net, criterion = trainerNet.getContrastiveDP(_WIDTH_TAB_[nsample], _DISP_TAB_[nsample],hpatch, opt['th_sup'],  opt['th_occ'],  opt['loss_margin'], _METRIC_NET_:clone():double())  
      else
        -- TO-DO
      end
    
      -- put training network on cuda
      tr_net:cuda()
      criterion:cuda()
      
      tr_param, tr_grad = tr_net:getParameters() 
    
    end
    
    tr_grad:zero()
    
    --_METRIC_NET_:forward({torch.rand(1,9,1242):cuda(), torch.rand(1,9,1242):cuda()})
    -- forward pass through net
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
  
    -- copy gradients to base net
    _METRIC_PGRAD_:add(tr_grad)
    
  end
 
  _TR_LOSS_ = _TR_LOSS_ / nb_comp_ttl
  _METRIC_PGRAD_:div(nb_comp_ttl);

  return _TR_LOSS_, _METRIC_PGRAD_
end

-- go through epoches
for nepoch = 1, opt['train_nb_epoch'] do

  local sample_loss= {}
  local start_time = os.time()
  
  -- go through batches
  for nbatch = 1, opt['train_nb_batch'] do
    
   -- get batch
   _TR_INPUT_, _WIDTH_TAB_, _DISP_TAB_ = unsupSet:get(opt['train_batch_size'])   
           
   -- some network need only two epopolar lines
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
   optim.adam(feval, _METRIC_PPARAM_, {}, _OPTIM_STATE_)    
   table.insert(sample_loss, _TR_LOSS_)
   
   collectgarbage()
   
  end

  local end_time = os.time()
  local time_diff = os.difftime(end_time,start_time);
  local timestamp = os.date("%Y_%m_%d_%X_")
    
  -- compute trainin loss for batch
  local train_loss = torch.Tensor(sample_loss):mean();
  
  -- run for validation subset and get :error
  local net_fname = 'tmp/' ..timestamp.. '.t7';
  
  local set_name
  if( set == 'kitti2015_ext' ) then
    set_name = 'kitti2015'
  elseif ( set == 'kitti_ext' ) then
    set_name = 'kitti';
  else
    set_name = set
  end
  local str = './main.lua ' .. set_name .. ' our_fast -a test_te -sm_terminate cnn -net_fname ../mil-mc-cnn/' .. net_fname 
  tmp = {};
  tmp[1] = _METRIC_NET_:clone():double()
  torch.save(net_fname, tmp, 'ascii');
  lfs.chdir('../mc-cnn')
  local handle = io.popen(str)
  local result = handle:read("*a")
  str_err = string.gsub(result,'\n','');
  valid_err = tonumber(str_err);
  lfs.chdir('../mil-mc-cnn')

  -- push new record to log
  local logRecord = {}
  logRecord['train_loss'] = train_loss
  logRecord['valid_err'] = valid_err
  logRecord['dt'] = time_diff
  table.insert(_TRAIN_LOG_, logRecord)
  
  -- save network
  local tmp = {}
  tmp[1] = _METRIC_NET_:clone():double()
  local optim_state = {}
  optim_state.t = _OPTIM_STATE_.t;
  optim_state.m = _OPTIM_STATE_.m:double();
  optim_state.v = _OPTIM_STATE_.v:double();
  optim_state.denom = _OPTIM_STATE_.denom:double();  
  tmp[2] = optim_state
  tmp[3] = _TRAIN_LOG_
  local net_fname = 'work/' .. opt['debug_fname'] .. '/metricNet_' .. opt['debug_fname'] .. timestamp.. '.t7';
  torch.save(net_fname, tmp, 'ascii');
  local net_fname = 'work/' .. opt['debug_fname'] .. '/metricNet_' .. opt['debug_fname'] .. '.t7';
  torch.save(net_fname, tmp, 'ascii');

  -- save log
  local log_fname = 'work/' .. opt['debug_fname'] .. '/err_' .. opt['debug_fname'] ..'_'.. timestampBeg.. '.txt';
  local f = io.open(log_fname, 'w')
  local time = 0
  for i = 1, #_TRAIN_LOG_ do
    time  = time  + _TRAIN_LOG_[i].dt
    f:write(string.format("%d, %f, %f, %f\n", i, time, _TRAIN_LOG_[i].train_loss, _TRAIN_LOG_[i].valid_err))
  end
  f:close()
  
  -- print log
  print(string.format("epoch %d, time = %f, train_loss = %f, valid_err = %f\n", nepoch, time, _TRAIN_LOG_[#_TRAIN_LOG_].train_loss, _TRAIN_LOG_[#_TRAIN_LOG_].valid_err))
  
collectgarbage()

end






