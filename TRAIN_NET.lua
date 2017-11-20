#!/usr/bin/env luajit

--[[ 

This is universal script for weekly-supervised training of network.

TRAIN_NET.lua <mode> <method> <arch> <set>

<mode>   - normal / debug.
<method> - mil / contrastive / mil-contrastive / contrastive-dp
<arch>   - fst-kitti / fst-mb
<set>    - kitti / kitti_ext / kitti15 / kitti15_ext / mb

]]--

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
cudnn.benchmark = true
cudnn.fastest = true

-- modules from mc-cnn 
include('../mc-cnn/Normalize2.lua')
include('../mc-cnn/StereoJoin1.lua')

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
testFun = dofile('CTestUtils.lua');       -- Function that performs test on validation set
utils = dofile('utils.lua');              -- Utils for loading and visualization

----------------------------------- parse input parameters -----------------------------------------------

dbg    = table.remove(arg, 1) 
method = table.remove(arg, 1)
arch   = table.remove(arg, 1)
set    = table.remove(arg, 1)

cmd = torch.CmdLine()

dbg    = dbg or 'debug';
method = method or 'contrastive-dp'
arch   = arch or 'fst-kitti'
set    = set or 'kitti'

assert( dbg == 'debug' or
        dbg == 'normal' )

assert( method == 'mil' or
        method == 'contrastive' or 
        method == 'mil-contrastive' or 
        method == 'contrastive-dp')

assert( arch == 'fst-mb' or 
        arch == 'fst-kitti')

assert( set == 'mb' or 
        set == 'kitti' or
        set == 'kitti2015' or
        set == 'kitti2015_ext' or
        set == 'kitti_ext')

if dbg == 'normal' then
  cmd:option('-train_batch_size', 300)   
  cmd:option('-train_nb_batch', 100)        
  cmd:option('-train_nb_epoch', 1000)        
else
  cmd:option('-train_batch_size', 32)     
  cmd:option('-train_nb_batch', 10)        
  cmd:option('-train_nb_epoch', 4)        
end

cmd:option('-loss_margin', 0.2)
cmd:option('-th_sup', 2) 
cmd:option('-th_occ', 1)    

if dbg == 'debug' then
  local timestamp = os.date("debug-%Y_%m_%d_%X")
  cmd:option('-debug_fname', timestamp)
end

cmd:option('-debug_start_from_net', '')
opt = cmd:parse(arg)
print('Weakly-suprevised training. method: ' .. method .. ', arch: ' .. arch.. ', set: ' .. set.. ', mode: ' .. dbg)

-------------------------------------- creat all nessesary output folders ---------------------------------- 

paths.mkdir('work/'..opt['debug_fname']); 
local timestampBeg = os.date("%Y_%m_%d_%X")
cmd:log('work/' .. opt['debug_fname'] .. '/log_' .. opt['debug_fname'] .. '_' .. timestampBeg .. '.txt', opt, 'ascii')

if dbg == 'debug' then
  math.randomseed(0); 
  torch.manualSeed(0)
end

---------------------------------------- initialize / load networks -----------------------------------------

local  hpatch
local _EMBED_NET_;
local _HEAD_NET_;

_EMBED_NET_ = cnnMetric.getEmbeddNet(arch)
_HEAD_NET_  = cnnMetric.getHeadNet(arch)
hpatch = cnnMetric.getHPatch(_EMBED_NET_)

_OPTIM_STATE_ = {}
_TRAIN_LOG_   = {}

if( opt.debug_start_from_net ~= '' ) then
  if utils.file_exists(opt.debug_start_from_net) then
    local tmp = torch.load(opt.debug_start_from_net, 'ascii')      
    _EMBED_NET_   = tmp[1]
    _HEAD_NET_    = tmp[2]
    _OPTIM_STATE_ = tmp[3]
    _TRAIN_LOG_   = tmp[4]
    print('Continue training from network specified by user\n')
  else
    error('Could not find network specified by user\n')
    _OPTIM_STATE_ = {}
  end
else
  local default_net = 'work/' .. opt['debug_fname'] .. '/metricNet_' .. opt['debug_fname'] .. '.t7'
  if utils.file_exists(default_net) then
    local tmp = torch.load(default_net, 'ascii')      
    _EMBED_NET_   = tmp[1]
    _HEAD_NET_    = tmp[2]
    _OPTIM_STATE_ = tmp[3]
    _TRAIN_LOG_   = tmp[4]
    print('Continue training from default network\n')
  else
    print('Could not find default network. Starting from the beggining.\n')
    _OPTIM_STATE_ = {}
  end
end

_HEAD_NET_:cuda()
_EMBED_NET_:cuda();
cudnn.convert(_HEAD_NET_, cudnn)
cudnn.convert(_EMBED_NET_, cudnn)

_EMBED_PARAM_, _EMBED_GRAD_ = _EMBED_NET_:getParameters() 
_HEAD_PARAM_, _HEAD_GRAD_   = _HEAD_NET_:getParameters() 

---------------------------- initialize dataset ------------------------------------------------------------

if set == 'kitti_ext' or set == 'kitti' then

  unsupSet = unsupKITTI('data/kitti_ext', set, hpatch);

elseif set == 'kitti2015' or set == 'kitti2015_ext' then

  unsupSet = unsupKITTI('data/kitti15_ext', set, hpatch);

else

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

----------------------------------------- define optimization function ------------------------------------

feval = function(param)

  -- set network parameters
  _EMBED_PARAM_:copy(param)
  
  local batch_size = #_TR_INPUT_[1];
  
  _TR_LOSS_ = 0;   

  _EMBED_GRAD_:zero()
  _HEAD_GRAD_:zero()

  local nb_comp_ttl = 0; 
  local criterion, tr_net

  for nsample = 1, batch_size do

    local sample_input = {}
    sample_input[1] = _TR_INPUT_[1][nsample]
    sample_input[2] = _TR_INPUT_[2][nsample]
    
    if _TR_INPUT_[3] ~= nil then
      sample_input[3] = _TR_INPUT_[3][nsample]
    end
    
    if( _WIDTH_TAB_[nsample] ~= _WIDTH_TAB_[nsample-1] or _DISP_MAX_TAB_[nsample] ~= _DISP_MAX_TAB_[nsample-1] )  then
  
      if method == 'mil' then

        tr_net, criterion = trainerNet.getMil(_DISP_MAX_TAB_[nsample], _WIDTH_TAB_[nsample], opt['loss_margin'], _EMBED_NET_, _HEAD_NET_)  

      elseif method == 'contrastive' then

        tr_net, criterion =  trainerNet.getContrastive(_DISP_MAX_TAB_[nsample], _WIDTH_TAB_[nsample], opt['th_occ'], opt['loss_margin'], _EMBED_NET_, _HEAD_NET_) 

      elseif method == 'mil-contrastive' then

        tr_net, criterion =  trainerNet.getMilContrastive(_DISP_MAX_TAB_[nsample], _WIDTH_TAB_[nsample], opt['th_sup'],  opt['th_occ'], opt['loss_margin'], _EMBED_NET_, _HEAD_NET_)

      else

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
  grad = _EMBED_GRAD_:clone()
  
  return _TR_LOSS_, grad
end

-------------------------------------- optimize --------------------------------------------------------
cur_param = _EMBED_PARAM_:clone();

local train_loss = 1/0;     
local time_per_epoch = 1/0 

for nepoch = 1, opt['train_nb_epoch'] do

  local start_time = os.time()
  local timestamp = os.date("%Y_%m_%d_%X")

  -- compute validation error using mc-cnn
  local WTA_err = 1/0
  local Pipeline_err = 1/0
  do
    temp_random_name = utils.get_random_file_name()
    temp_net_name = 'tmp/' .. temp_random_name .. '.t7';

    local temp_network = {};
    temp_network[1] = _EMBED_NET_:clone()
    temp_network[1] = cnnMetric.padBoundary(temp_network[1])
    temp_network[1]:add( nn.Normalize2():cuda() )
    temp_network[1]:add( nn.StereoJoin1():cuda() )
    
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
    lfs.chdir('../mc-cnn') 
    
    -- WTA Err
    local temp_exec_str 
    temp_exec_str = './main.lua ' .. temp_set_name .. ' fast -a test_te -sm_terminate cnn  -net_fname ../mil-mc-cnn/' .. temp_net_name 
    local temp_handle = io.popen(temp_exec_str)
    local temp_result = temp_handle:read("*a")
    local temp_valid_err_str = string.match(temp_result, '\n0.%d+\n$') --string.gsub(temp_result,'\n','');
    WTA_err = tonumber(temp_valid_err_str);

    -- Pipeline Err
    temp_exec_str = './main.lua ' .. temp_set_name .. ' fast -a test_te -net_fname ../mil-mc-cnn/' .. temp_net_name 
    temp_handle = io.popen(temp_exec_str)
    temp_result = temp_handle:read("*a")
    temp_valid_err_str = string.match(temp_result, '\n0.%d+\n$')--string.gsub(temp_result,'\n','');
    Pipeline_err = tonumber(temp_valid_err_str);

    lfs.chdir('../mil-mc-cnn') -- switch current direc-- decrease 
    cutorch.synchronize()
  end

  -- add records to training log
  do
    local temp_trainingLog = {}
    temp_trainingLog['train_loss']   = train_loss
    temp_trainingLog['WTA_err']      = WTA_err
    temp_trainingLog['Pipeline_err'] = Pipeline_err
    table.insert(_TRAIN_LOG_, temp_trainingLog)
  end

  -- save current network and optimization state
  local cur_net_name 
  do

    local temp_network = {}
    temp_network[1] = _EMBED_NET_
    temp_network[2] = _HEAD_NET_
    temp_network[3] = _OPTIM_STATE_
    temp_network[4] = _TRAIN_LOG_

    local temp_hist_net_name = 'work/' .. opt['debug_fname'] .. '/metricNet_' .. opt['debug_fname'] .. '_' .. timestamp.. '.t7';  
    torch.save(temp_hist_net_name, temp_network, 'ascii');
    cur_net_name = 'work/' .. opt['debug_fname'] .. '/metricNet_' .. opt['debug_fname'] .. '.t7';  -- current network log 
    torch.save(cur_net_name , temp_network, 'ascii');

  end

  -- save training log
  local log_fname 
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

  -- print optimization results
  print(string.format("epoch %d, time_per_epoch %f, train_loss = %f , WTA_err = %f , Pipeline_err  = %f \n", nepoch, time_per_epoch, 
      _TRAIN_LOG_[#_TRAIN_LOG_].train_loss, 
      _TRAIN_LOG_[#_TRAIN_LOG_].WTA_err,
      _TRAIN_LOG_[#_TRAIN_LOG_].Pipeline_err))

  -- save optimization plot
  do
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
    gnuplot.raw("plot '" .. log_fname .. "' using ($1):(($3))  title 'WTA err' axes x1y2, '" .. log_fname .. "' using ($1):(($4))  title 'Pipeline err' axes x1y2, '" .. log_fname .. "' using ($1):(($2))  title 'Train loss' axes x1y1")
    gnuplot.plotflush()
  end  

  -- optimize
  collectgarbage()
  local batch_loss = {}
  for nbatch = 1, opt['train_nb_batch'] do

    _TR_INPUT_, _WIDTH_TAB_, _DISP_MAX_TAB_ = unsupSet:get(opt['train_batch_size'], '../mil-mc-cnn/' .. cur_net_name )   

    if method == 'contrastive' or method == 'contrastive-dp' then
      _TR_INPUT_[3] = nil
    end

    for nsample = 1, opt.train_batch_size do
      for i = 1,#_TR_INPUT_ do
        _TR_INPUT_[i][nsample] = _TR_INPUT_[i][nsample]:cuda()
      end
    end

    config = {}
    config.learningRate = 1e-4
    optim.adam(feval, cur_param, config, _OPTIM_STATE_)    
    batch_loss[nbatch] = _TR_LOSS_

    _EMBED_PARAM_:copy(cur_param)
    
    _EMBED_NET_:zeroGradParameters()
    _HEAD_NET_:zeroGradParameters()

    collectgarbage()
  end

  -- compute time
  local end_time = os.time()
  time_per_epoch = os.difftime(end_time, start_time);
  
  -- compute trainin loss for epoch
  train_loss = torch.Tensor(batch_loss):mean();
  
end

----------------------------------------------- save final network in mc-cnn compatible form ---------------------------
local final_net_name = 'work/' .. opt['debug_fname'] .. '/metricNet_' .. opt['debug_fname'] .. '_FINAL.t7';
local temp_network = {};
temp_network[1] = _EMBED_NET_:clone()
temp_network[1] = cnnMetric.padBoundary(temp_network[1])
temp_network[1]:add( nn.Normalize2():cuda() )
temp_network[1]:add( nn.StereoJoin1():cuda() )
    
torch.save(final_net_name, temp_network, 'ascii');





