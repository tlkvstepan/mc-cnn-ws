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
require 'os'

-- cuda 
require 'cunn'
require 'cudnn'
-- this configuration seems fastest
cudnn.benchmark = true
--cudnn.fastest = true

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
dofile('CPipeline.lua');

dofile('CUnsupMB.lua')                     
dofile('CUnsupPipeMB.lua')  
dofile('CUnsupKITTI.lua')
dofile('CUnsupPipeKITTI_with_GT.lua')

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
method = method or 'pipeline'
arch = arch or 'fst-mb'
set = set or 'mb'

assert(method == 'mil' or method == 'contrastive' or method == 'mil-contrastive' or method == 'contrastive-dp' or method == 'pipeline')
assert(arch == 'fst-mb' or arch == 'fst-kitti' or arch == 'fst-kitti-4x' or arch == 'acrt-mb' or arch == 'acrt-kitti' or arch == 'fst-xxl')
assert(set == 'mb' or set == 'kitti' or set == 'kitti2015' or set == 'kitti2015_ext' or set == 'kitti_ext')

cmd:option('-use_gt', 0) -- dont use GT by default  


if dbg == 'normal' then
  -- for real training 
  cmd:option('-train_batch_size', 300)   
  cmd:option('-train_nb_batch', 10)        
  cmd:option('-train_nb_epoch', 1000)        
elseif dbg == 'tune' then
  cmd:option('-train_batch_size', 300)     
  cmd:option('-train_nb_batch', 5)        
  cmd:option('-train_nb_epoch', 10)        
else 
  cmd:option('-train_batch_size', 300)     
  cmd:option('-train_nb_batch', 1)        
  cmd:option('-train_nb_epoch', 10)        
end

-- semi-supervised method parameters
cmd:option('-loss_margin', 0.2)

-- for pipeline we 
if method == 'pipeline' then
  cmd:option('-th_sup', 20) --20 
else
  cmd:option('-th_sup', 2) 
end
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

if dbg == 'debug' then
  math.randomseed(0); 
  torch.manualSeed(0)
end
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
    _OPTIM_STATE_ = {}
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
    _OPTIM_STATE_ = {}
  end
end

--if opt.reset_optim == 1 then
--    _OPTIM_STATE_ = {}
--end

-- put optimization state on gpu
--if _OPTIM_STATE_.m then
--  _OPTIM_STATE_.m = _OPTIM_STATE_.m:cuda()
--  _OPTIM_STATE_.v = _OPTIM_STATE_.v:cuda()
--  _OPTIM_STATE_.denom = _OPTIM_STATE_.denom:cuda()
--end

_HEAD_NET_:cuda()
_EMBED_NET_:cuda();
cudnn.convert(_HEAD_NET_, cudnn)
cudnn.convert(_EMBED_NET_, cudnn)

_EMBED_PARAM_, _EMBED_GRAD_ = _EMBED_NET_:getParameters() 
_HEAD_PARAM_, _HEAD_GRAD_   = _HEAD_NET_:getParameters() 

---------------------------- initialize dataset

if set == 'kitti_ext' or set == 'kitti' then

  if method == 'pipeline' then

    unsupSet = unsupPipeKITTI('data/kitti_ext', set, opt.use_gt, hpatch, opt.debug_fname);
    --unsupSet = unsupPipeKITTI('data/kitti_ext', set, hpatch);

  else

    unsupSet = unsupKITTI('data/kitti_ext', set, hpatch);

  end

elseif set == 'kitti2015' or set == 'kitti2015_ext' then

  if method == 'pipeline' then  

    unsupSet = unsupPipeKITTI('data/kitti15_ext', set, opt.use_gt, hpatch, opt.debug_fname);
    --unsupSet = unsupPipeKITTI('data/kitti15_ext', set, hpatch);

  else

    unsupSet = unsupKITTI('data/kitti15_ext', set, hpatch);

  end

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

  if method == 'pipeline' then

   --error('training with pipeline for mb is not yet implemented')
    unsupSet = unsupPipeMB(img_tab, metadata, hpatch, opt.debug_fname)

  else

    unsupSet = unsupMB(img_tab, metadata, hpatch)

  end

end


-- |define optimization function|
feval = function(param)

  -- set network parameters
  if arch == 'fst-kitti' or arch == 'fst-kitti-4x' or arch == 'fst-mb' or arch == 'fst-xxl' then
    _EMBED_PARAM_:copy(param)
  else
    _EMBED_PARAM_:copy(param[{{1, _EMBED_PARAM_:size(1)}}])
    _HEAD_PARAM_:copy(param[{{1+_EMBED_PARAM_:size(1), param:size(1)}}])
  end

  local batch_size = #_TR_INPUT_[1];
  --local epi, match = unpack(_TR_INPUT_) -- epiNeg does not exist for  contrastive-max and contrastive-dprog
  --local epiRef, epiPos, epiNeg = unpack(epi) -- epiNeg does not exist for  contrastive-max and contrastive-dprog

  _TR_LOSS_ = 0;   

  -- clear gradients
  _EMBED_GRAD_:zero()
  _HEAD_GRAD_:zero()

  local nb_comp_ttl = 0; 
  local criterion, tr_net

  for nsample = 1, batch_size do

    -- print(nsample)

    -- if nsample == 51 then
    --   i = 1;
    -- end

    local sample_input = {}

    if ( method == 'pipeline' ) then 
      sample_input[1] = {}
      sample_input[1][1] = _TR_INPUT_[1][nsample]
      sample_input[1][2] = _TR_INPUT_[2][nsample]
      sample_input[2] = _TR_INPUT_[3][nsample]
    else
      sample_input[1] = _TR_INPUT_[1][nsample]
      sample_input[2] = _TR_INPUT_[2][nsample]
      if _TR_INPUT_[3] ~= nil then
        sample_input[3] = _TR_INPUT_[3][nsample]
      end
    end
    -- if epiNeg ~= nil then
    --   sample_input[3] = epiNeg[nsample]
    -- end

    if( _WIDTH_TAB_[nsample] ~= _WIDTH_TAB_[nsample-1] or _DISP_MAX_TAB_[nsample] ~= _DISP_MAX_TAB_[nsample-1] )  then

      if method == 'mil' then

        tr_net, criterion = trainerNet.getMil(_DISP_MAX_TAB_[nsample], _WIDTH_TAB_[nsample], opt['loss_margin'], _EMBED_NET_, _HEAD_NET_)  

      elseif method == 'contrastive' then

        tr_net, criterion =  trainerNet.getContrastive(_DISP_MAX_TAB_[nsample], _WIDTH_TAB_[nsample], opt['th_occ'], opt['loss_margin'], _EMBED_NET_, _HEAD_NET_) 

      elseif method == 'mil-contrastive' then

        tr_net, criterion =  trainerNet.getMilContrastive(_DISP_MAX_TAB_[nsample], _WIDTH_TAB_[nsample], opt['th_sup'],  opt['th_occ'], opt['loss_margin'], _EMBED_NET_, _HEAD_NET_)

      elseif( method == 'contrastive-dp' ) then

        tr_net, criterion =  trainerNet.getContrastiveDP(_DISP_MAX_TAB_[nsample], _WIDTH_TAB_[nsample], opt['th_sup'],  opt['th_occ'],  opt['loss_margin'], _EMBED_NET_, _HEAD_NET_)  

      elseif( method == 'pipeline' ) then 

        tr_net, criterion = trainerNet.getPipeline(_DISP_MAX_TAB_[nsample], _WIDTH_TAB_[nsample], opt['th_sup'],  opt['loss_margin'], _EMBED_NET_, _HEAD_NET_)

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

  if arch == 'fst-kitti' or arch == 'fst-mb' or arch == 'fst-xxl' or  arch == 'fst-kitti-4x' then
    grad = _EMBED_GRAD_:clone()
  else 
    _HEAD_GRAD_:div(nb_comp_ttl);
    grad = torch.cat(_EMBED_GRAD_, _HEAD_GRAD_, 1)
  end

  return _TR_LOSS_, grad
end

-- initialize the parameters
if arch == 'fst-kitti' or  arch == 'fst-kitti-4x' or arch == 'fst-mb' or arch == 'fst-xxl' then
  cur_param = _EMBED_PARAM_:clone();
else
  cur_param = torch.cat(_EMBED_PARAM_, _HEAD_PARAM_, 1)
end


local train_loss = 1/0;     
local time_per_epoch = 1/0 
for nepoch = 1, opt['train_nb_epoch'] do

  local start_time = os.time()
  local timestamp = os.date("%Y_%m_%d_%X_")

  -- COMPUTE VALIDATION ERROR
  local WTA_err = 1/0
  local Pipeline_err = 1/0
  do
    temp_random_name = utils.get_random_file_name()
    temp_net_name = 'tmp/' .. temp_random_name .. '.t7';

    -- save current network 
    local temp_network = {};
    temp_network[1] = _EMBED_NET_
    temp_network[2] = _HEAD_NET_
    torch.save(temp_net_name, temp_network, 'ascii');

    local temp_set_name
    if  ( set == 'kitti2015_ext' ) then 
      temp_set_name = 'kitti2015'
    elseif ( set == 'kitti_ext' ) then 
      temp_set_name = 'kitti';
    else
      temp_set_name = set 
    end

    local temp_exec_str
    lfs.chdir('../mc-cnn')  -- switch current directory
    local temp_test_size
    if dbg == 'normal' then 
      temp_test_size = 0 -- use all test images for test
    else
      temp_test_size = 5 -- use only 5 images for test
    end

    -- compute WTA Err
    local temp_exec_str 
    temp_exec_str = './main.lua ' .. temp_set_name .. ' our-fast -a test_te -test_size '.. temp_test_size ..
    ' -sm_terminate cnn  -net_fname ../mil-mc-cnn/' .. temp_net_name 
    local temp_handle = io.popen(temp_exec_str)
    local temp_result = temp_handle:read("*a")
    local temp_valid_err_str = string.gsub(temp_result,'\n','');
    WTA_err = tonumber(temp_valid_err_str);

    -- compute Pipeline Err
    temp_exec_str = './main.lua ' .. temp_set_name .. ' our-fast -a test_te -test_size '.. temp_test_size ..
    ' -net_fname ../mil-mc-cnn/' .. temp_net_name 
    temp_handle = io.popen(temp_exec_str)
    temp_result = temp_handle:read("*a")
    temp_valid_err_str = string.gsub(temp_result,'\n','');
    Pipeline_err = tonumber(temp_valid_err_str);

    lfs.chdir('../mil-mc-cnn') -- switch current directory back  
    cutorch.synchronize()
  end

  -- ADD RECORDS TO TRAINING LOG
  do
    local temp_trainingLog = {}
    temp_trainingLog['train_loss']   = train_loss
    temp_trainingLog['WTA_err']      = WTA_err
    temp_trainingLog['Pipeline_err'] = Pipeline_err
    table.insert(_TRAIN_LOG_, temp_trainingLog)
  end

  -- SAVE CURRENT NETWORK WITH TRAINING LOG AND OPTIMIZATION STATE
  local cur_net_name -- need it for saving after each batch
  do

    local temp_network = {}
    temp_network[1] = _EMBED_NET_
    temp_network[2] = _HEAD_NET_
    temp_network[3] = _OPTIM_STATE_
    temp_network[4] = _TRAIN_LOG_

    local temp_hist_net_name = 'work/' .. opt['debug_fname'] .. '/metricNet_' .. opt['debug_fname'] .. timestamp.. '.t7'; -- history network log 
    torch.save(temp_hist_net_name, temp_network, 'ascii');
    cur_net_name = 'work/' .. opt['debug_fname'] .. '/metricNet_' .. opt['debug_fname'] .. '.t7';  -- current network log 
    torch.save(cur_net_name , temp_network, 'ascii');

  end

  -- SAVE TEXT LOG FILE
  local log_fname -- need it for graph
  do
    log_fname = 'work/' .. opt['debug_fname'] .. '/err_' .. opt['debug_fname'] ..'_'.. timestampBeg.. '.txt';
    local temp_file = io.open(log_fname, 'w')
    temp_file:write(string.format("%f, %f, %f, %f\n", 1, 1/0, _TRAIN_LOG_[1].WTA_err, _TRAIN_LOG_[1].Pipeline_err)) 
    for i = 2, #_TRAIN_LOG_ do
      if _TRAIN_LOG_[i].train_loss ~= 1/0 then 
        -- we skip all other records with inf training loss, except of the first one
        temp_file:write(string.format("%f, %f, %f, %f\n", i, _TRAIN_LOG_[i].train_loss, _TRAIN_LOG_[i].WTA_err, _TRAIN_LOG_[i].Pipeline_err))
      end
    end
    temp_file:close()
  end

  -- COMPUTE AVERAGE SPEED OF ERROR CHANGES
  local Avg_Pipeline_Err_Spd = {}
  local Avg_WTA_Err_Spd = {}
  local Avg_Train_Loss_Spd = {}
  do
    local Pipeline_Err_Spd = {}
    local WTA_Err_Spd = {}
    local Train_Loss_Spd = {}
    k = 1;
    for i = #_TRAIN_LOG_, 1, -1 do
      if _TRAIN_LOG_[i].train_loss ~= 1/0 then
        -- we skip all other records with inf training loss, except of the first one
        Pipeline_Err_Spd[k] = _TRAIN_LOG_[i].Pipeline_err 
        WTA_Err_Spd[k]      = _TRAIN_LOG_[i].WTA_err 
        Train_Loss_Spd[k]   = _TRAIN_LOG_[i].train_loss 
        k = k + 1
        if( k > 10 ) then 
          break;
        end
      end
    end
    if( k > 2 ) then
      for i = 1,#Pipeline_Err_Spd-1 do
        Pipeline_Err_Spd[i] = Pipeline_Err_Spd[i]-Pipeline_Err_Spd[i+1]
        WTA_Err_Spd[i] = WTA_Err_Spd[i]-WTA_Err_Spd[i+1]
        Train_Loss_Spd[i] = Train_Loss_Spd[i]-Train_Loss_Spd[i+1]
      end
      Avg_Train_Loss_Spd = torch.Tensor(Train_Loss_Spd)[{{1,-2}}]:mean()
      Avg_WTA_Err_Spd = torch.Tensor(WTA_Err_Spd)[{{1,-2}}]:mean()
      Avg_Pipeline_Err_Spd = torch.Tensor(Pipeline_Err_Spd)[{{1,-2}}]:mean()
    else
      Avg_Pipeline_Err_Spd = 0;
      Avg_WTA_Err_Spd = 0
      Avg_Train_Loss_Spd = 0
    end
  end  

  -- PRINT LOG
  print(string.format("epoch %d, time_per_epoch %f, train_loss = %f (%f), WTA_err = %f (%f), Pipeline_err  = %f (%f)\n", nepoch, time_per_epoch, 
      _TRAIN_LOG_[#_TRAIN_LOG_].train_loss, Avg_Train_Loss_Spd, 
      _TRAIN_LOG_[#_TRAIN_LOG_].WTA_err, Avg_WTA_Err_Spd,
      _TRAIN_LOG_[#_TRAIN_LOG_].Pipeline_err, Avg_Pipeline_Err_Spd))

  -- SAVE PLOT
  do
    -- two graphs, one for training (blue) and one for test (red)
    gnuplot.epsfigure('work/' .. opt['debug_fname'] .. '/plot_' .. opt['debug_fname'] .. '.eps')
    gnuplot.raw('set ylabel "train loss"')
    gnuplot.raw('set y2label "valid. err, [%]"')
    gnuplot.raw('set xlabel "iter"')
    gnuplot.raw('set x1tic auto')
    gnuplot.raw('set y1tic auto')
    gnuplot.raw('set y2tic auto')
    gnuplot.raw('set logscale x')
    gnuplot.raw('set logscale y')
    gnuplot.raw('set logscale y2')
    gnuplot.raw("plot '" .. log_fname .. "' using ($1):(($3))  title 'WTA err' axes x1y2, '" .. log_fname .. "' using ($1):(($4))  title 'Pipeline err' axes x1y2, '" .. log_fname .. "' using ($1):(($2))  title 'train. loss' axes x1y1")
    gnuplot.plotflush()
  end  

  -- GO THROUGH BATCHES
  collectgarbage()
  local batch_loss = {}
  for nbatch = 1, opt['train_nb_batch'] do

    -- get batch
    _TR_INPUT_, _WIDTH_TAB_, _DISP_MAX_TAB_ = unsupSet:get(opt['train_batch_size'], '../mil-mc-cnn/' .. cur_net_name )   

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
    config = {}
    config.learningRate = 1e-4 
    optim.adam(feval, cur_param, config, _OPTIM_STATE_)    
    batch_loss[nbatch] = _TR_LOSS_

    -- update parameters
    if arch == 'fst-kitti' or arch == 'fst-kitti-4x' or arch == 'fst-mb' or arch == 'fst-xxl' then
      _EMBED_PARAM_:copy(cur_param)
    else
      _EMBED_PARAM_:copy(cur_param[{{1, _EMBED_PARAM_:size(1)}}])
      _HEAD_PARAM_:copy(cur_param[{{1+_EMBED_PARAM_:size(1), cur_param:size(1)}}])
    end


    -- clean gradient
    _EMBED_NET_:zeroGradParameters()
    _HEAD_NET_:zeroGradParameters()

    -- save net 
    do 
      local temp_network = {};
      temp_network[1] = _EMBED_NET_
      temp_network[2] = _HEAD_NET_
      temp_network[3] = _OPTIM_STATE_
      temp_network[4] = _TRAIN_LOG_
      torch.save(cur_net_name, temp_network, 'ascii');
    end

    collectgarbage()
  end

  -- compute time
  local end_time = os.time()
  time_per_epoch = os.difftime(end_time, start_time);
  
  -- compute trainin loss for epoch
  train_loss = torch.Tensor(batch_loss):mean();

  -- decrease 
  if method == 'pipeline' and dbg == 'normal' then
    opt.th_sup = math.max(torch.round(opt.th_sup / 1.3), 2)
  end

end






