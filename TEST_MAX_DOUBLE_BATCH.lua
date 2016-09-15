require 'torch'
require 'gnuplot'
require 'optim'
require 'nn'

dofile('DataLoader.lua');
dofile('CUnsup3EpiSet.lua');
dofile('CSup3PatchSet.lua');
mcCnnFst = dofile('CMcCnnFst.lua');
dofile('CAddMatrix.lua')
dofile('CMaxM.lua')
milWrapper = dofile('CMilWrapper.lua')
utils = dofile('utils.lua');

-- set randomseed to insure repeatability
math.randomseed(0); 
torch.manualSeed(0)

-- |parse input parameters|
cmd = torch.CmdLine()
-- learning
cmd:option('-test_set_size', 200000)      -- 200000
cmd:option('-train_batch_size', 1024)     -- 1024
cmd:option('-train_epoch_size', 100*1024) -- 100*1024
cmd:option('-train_nb_epoch', 300)
-- loss
cmd:option('-loss_margin', 0.2)
cmd:option('-max_order', 2)
cmd:option('-max_r', 2)
-- network
cmd:option('-net_nb_feature', 64)
cmd:option('-net_kernel', 3)
cmd:option('-net_nb_layers', 4)
-- debug
cmd:option('-debug_fname', 'maxLargeScale_maxsup')
cmd:option('-debug_gpu_on', true)
cmd:option('-debug_save_on', true)
cmd:option('-debug_only_test_on', false)
cmd:option('-debug_start_from_timestamp', '')
prm = cmd:parse(arg)
paths.mkdir('work/'..prm['debug_fname']); -- make output folder

print('MIL training started \n')
print('Parameters of the procedure : \n')
utils.printTable(prm)

if( prm['debug_gpu_on'] ) then
  require 'cunn'
end

-- |read data| from all KITTI
local img1_arr = torch.squeeze(utils.fromfile('data/KITTI12/x0.bin'));
local img2_arr = torch.squeeze(utils.fromfile('data/KITTI12/x1.bin'));
local disp_arr = torch.round(torch.squeeze(utils.fromfile('data/KITTI12/dispnoc.bin')));

local disp_max = disp_arr:max()
local img_w = img1_arr:size(3);

-- |define test and training networks|
local base_fnet
local hpatch
local fnet_fname = 'work/' .. prm['debug_fname'] .. '/fnet_' .. prm['debug_start_from_timestamp'] .. '_' .. prm['debug_fname'] .. '.t7'
local optim_fname = 'work/' .. prm['debug_fname'] .. '/optim_' .. prm['debug_start_from_timestamp'] .. '_'.. prm['debug_fname'] .. '.t7'
if utils.file_exists(fnet_fname) and utils.file_exists(optim_fname) then
  print('Continue training. Please delete the network file if you wish to start from the beggining\n')
  _BASE_FNET_= torch.load(fnet_fname, 'ascii')
  hpatch = ( utils.get_window_size(_BASE_FNET_)-1 )/ 2
  _OPTIM_STATE_ =  torch.load(optim_fname, 'ascii')
else
  print('Start training from the begining\n')
  _BASE_FNET_, hpatch = mcCnnFst.get(prm['net_nb_layers'], prm['net_nb_feature'], prm['net_kernel'])
  _OPTIM_STATE_ = {}
end

_TR_NET_ =  milWrapper.getMaxNetDoubleBatch(img_w, disp_max, hpatch, prm['max_order'], prm['max_r'], _BASE_FNET_)  
_TE_NET_ = milWrapper.getTripletNet(_BASE_FNET_) 

if prm['debug_gpu_on'] then
  _TR_NET_:cuda()
  _TE_NET_:cuda()
  if _OPTIM_STATE_.m then
    _OPTIM_STATE_.m = _OPTIM_STATE_.m:cuda()
    _OPTIM_STATE_.v = _OPTIM_STATE_.v:cuda()
    _OPTIM_STATE_.denom = _OPTIM_STATE_.denom:cuda()
  end
end

_BASE_PPARAM_ = _BASE_FNET_:getParameters() 
_TR_PPARAM_, _TR_PGRAD_ = _TR_NET_:getParameters()
_TE_PPARAM_, _TE_PGRAD_ = _TE_NET_:getParameters()

-- |define datasets|
local trainSet = unsup3EpiSet(img1_arr, img2_arr, hpatch, disp_max);
local testSet = sup3PatchSet(img1_arr[{{1,194},{},{}}], img2_arr[{{1,194},{},{}}], disp_arr[{{1,194},{},{}}], hpatch);
testSet:shuffle() -- so that we have patches from all images

-- |prepare test set|
_TE_INPUT_, target = testSet:index(torch.range(1, prm['test_set_size']))
_TE_TARGET_ = torch.ones(prm['test_set_size'])
if prm['debug_gpu_on'] then
  _TE_TARGET_ = _TE_TARGET_:cuda()
  _TE_INPUT_[1] = _TE_INPUT_[1]:cuda();
  _TE_INPUT_[2] = _TE_INPUT_[2]:cuda();
  _TE_INPUT_[3] = _TE_INPUT_[3]:cuda();
end

-- |define criterion|
-- loss(x(+), x(-)) = max(0,  - x(+) + x(-)  + prm['loss_margin'])
_CRITERION_ = nn.MarginRankingCriterion(prm['loss_margin']);
if prm['debug_gpu_on'] then
  _CRITERION_:cuda()
end

-- |define  test function|
ftest = function()

  -- function returns test accuracy and mask of errors

  _TE_PPARAM_:copy(_TR_PPARAM_)

  local nb_samples = _TE_INPUT_[1]:size(1);
  local patchRef, patchPos, patchNeg, dispDiff = unpack(_TE_INPUT_) 

  local err_mask = {};
  for nsample = 1, nb_samples do

    local sample_input = {{patchPos[{{nsample},{},{}}], patchNeg[{{nsample},{},{}}]}, patchRef[{{nsample},{},{}}]}

    -- forward pass
    local out = _TE_NET_:forward(sample_input)
    local pos_cos = torch.squeeze(out[1]:double())
    local neg_cos = torch.squeeze(out[2]:double())

    if pos_cos <= neg_cos then
      err_mask[nsample] = 1;
    else
      err_mask[nsample] = 0;  
    end

  end

  local err_mask = torch.Tensor(err_mask)

  local acc = err_mask[err_mask:eq(0)]:numel() * 100 / nb_samples

  local mask_le2 = dispDiff:le(2)
  local acc_le2 = err_mask[torch.cmul(mask_le2,err_mask:eq(0))]:numel() * 100 / err_mask[mask_le2]:numel(); 

  local mask_le5 = dispDiff:le(5);
  local acc_le5 = err_mask[torch.cmul(mask_le5,err_mask:eq(0))]:numel() * 100 / err_mask[mask_le5]:numel(); 

  local err_index = torch.range(1,nb_samples);
  local err_index = err_index[err_mask:eq(1)]

  return acc, acc_le2, acc_le5, err_index, dispDiff[err_mask:eq(1)]    
end

-- |define optimization function|
feval = function(x)

  -- set net parameters
  _TR_PPARAM_:copy(x)

  -- clear gradients
  _TR_PGRAD_:zero()

  local batch_size = _TR_INPUT_[1]:size(1);
  local epiRef, epiPos, epiNeg = unpack(_TR_INPUT_) 

  _TR_ERR_ = 0;   
  for nsample = 1, batch_size do

    local sample_input = {epiRef[{{nsample},{},{}}], epiPos[{{nsample},{},{}}]}
    local sample_target = _TR_TARGET_[{{nsample},{}}]    

    -- forward pass
    _TR_ERR_ = _TR_ERR_ + _CRITERION_:forward(_TR_NET_:forward(sample_input), sample_target)

    -- backword pass
    _TR_NET_:backward(sample_input, _CRITERION_:backward(_TR_NET_.output, sample_target))

  end
  _TR_ERR_ = _TR_ERR_ / prm['train_batch_size'] / 2
  _TR_PGRAD_:div(2*prm['train_batch_size']);

  return _TR_ERR_, _TR_PGRAD_      
end

-- |save debug info|
if prm['debug_save_on'] then
  
  -- save train parameters
  local timestamp = os.date("%Y_%m_%d_%X_")
  torch.save('work/' .. prm['debug_fname'] .. '/params_' .. timestamp .. prm['debug_fname'] .. '.t7', prm, 'ascii');
  
  --save test set statistic
  gnuplot.epsfigure('work/' .. prm['debug_fname'] .. '/testset_stat_' .. timestamp .. prm['debug_fname'] .. '.eps')
  gnuplot.hist(_TE_INPUT_[4], disp_max)
  gnuplot.xlabel('Distance between positive and negative example')
  gnuplot.ylabel('Number of samples')
  gnuplot.plotflush()
end
    
-- |define logger|
if prm['debug_save_on'] then
  logger = optim.Logger('work/' .. prm['debug_fname'] .. '/'.. prm['debug_fname'], true)
  logger:setNames{'Training loss', 
    'Overall test accuracy', 
    'Test accuracy 1-2 px',
    'Test accuracy 1-5 px'}
  logger:style{'+-',
    '+-',
    '+-',
    '+-'}
end    

-- |optimize network|   
local start_time = os.time()
for nepoch = 1, prm['train_nb_epoch'] do

  nsample = 0;
  sample_err = {}
  local train_err = 0
   
 
   for k, input  in trainSet:sampleiter(prm['train_batch_size'], prm['train_epoch_size']) do

   _TR_INPUT_ = input
   _TR_TARGET_ =  torch.ones(prm['train_batch_size'], 2*(img_w - disp_max - 2*hpatch));  


   -- if gpu avaliable put batch on gpu
   if prm['debug_gpu_on'] then
     _TR_INPUT_[1] = _TR_INPUT_[1]:cuda()
     _TR_INPUT_[2] = _TR_INPUT_[2]:cuda()
     _TR_INPUT_[3] = _TR_INPUT_[3]:cuda()
     _TR_TARGET_ = _TR_TARGET_:cuda()
   end
    
   -- if test mode, dont do training 
   if not prm['debug_only_test_on'] then
     optim.adam(feval, _TR_PPARAM_, {}, _OPTIM_STATE_)    
   else 
     feval(_TR_PPARAM_)
   end

   table.insert(sample_err, _TR_ERR_)

 end
 train_err = torch.Tensor(sample_err):mean();

  -- validation
  local test_acc, test_acc_le2, test_acc_le5, err_index, disp_diff  = ftest()
  
  local end_time = os.time()
  local time_diff = os.difftime(end_time,start_time);

  -- save prm['debug_save_on'] info
  if prm['debug_save_on'] then
    local timestamp = os.date("%Y_%m_%d_%X_")
   
    -- save errorneous test samples
    local fail_img = utils.vis_errors(_TE_INPUT_[1]:float():clone(), 
      _TE_INPUT_[2]:float():clone(), 
      _TE_INPUT_[3]:float():clone(), err_index, disp_diff)
    image.save('work/' .. prm['debug_fname'] .. '/error_cases_' .. timestamp .. prm['debug_fname'] .. '.png',fail_img)
    
    -- save net
    _BASE_PPARAM_:copy(_TR_PPARAM_)
    torch.save('work/' .. prm['debug_fname'] .. '/fnet_' .. timestamp .. prm['debug_fname'] .. '.t7', _BASE_FNET_, 'ascii');
    
    -- save optim state
    local optim_state_host ={}
    optim_state_host.t = _OPTIM_STATE_.t;
    optim_state_host.m = _OPTIM_STATE_.m:clone():double();
    optim_state_host.v = _OPTIM_STATE_.v:clone():double();
    optim_state_host.denom = _OPTIM_STATE_.denom:clone():double();
    torch.save('work/' .. prm['debug_fname'] .. '/optim_'.. timestamp .. prm['debug_fname'] .. '.t7', optim_state_host, 'ascii');
        
    -- save log
    logger:add{train_err, test_acc, test_acc_le2, test_acc_le5}
    logger:plot()
    
    -- save distance matrices
    local lines = {290,433}
    for nline = 1,#lines do
      input = trainSet:index(torch.Tensor{lines[nline]})
      _TR_NET_:forward({input[1]:cuda(),input[2]:cuda(),input[3]:cuda()} )
      refPos = _TR_NET_:get(2).output:clone():float();
      refPos = utils.mask(refPos,disp_max)
      refPos = utils.softmax(refPos)
      refPos = utils.scale2_01(refPos)
      image.save('work/' .. prm['debug_fname'] .. '/dist_' ..  string.format("line%i_",lines[nline])  .. timestamp .. prm['debug_fname'] .. '.png',refPos)
    end
  end
  
  print(string.format("epoch %d, time = %f, train_err = %f, test_acc = %f", nepoch, time_diff, train_err, test_acc))
  collectgarbage()

end





