require 'torch'
require 'gnuplot'
require 'nn'
require 'optim'
require 'cunn'

--require 'cutorch';

local dl = require 'dataload'
dofile('CUnsup3EpiSet.lua');
dofile('CSup3PatchSet.lua');
dofile('CMcCnnFst.lua');
dofile('CAddMatrix.lua')

utils = dofile('utils.lua');

-- |parameteres|
-- learning
local test_set_size = 10000;
local batch_size = 32
local epoch_size = batch_size*1000
local nb_epoch = 15
-- loss
local margin = 0.2;
-- network
nbFeatureMap = 64;
kernel = 3;
nbConvLayers = 5;
-- debug
local subset = 194
local gpu_on = true;
local graph_on = true;

-- |read data|
local img1_arr = torch.squeeze(utils.fromfile('x0.bin')):double();
local img2_arr = torch.squeeze(utils.fromfile('x1.bin')):double();
local disp_arr = torch.round(torch.squeeze(utils.fromfile('dispnoc.bin'))):double();
local img1_arr = img1_arr[{{1,subset},{},{}}]
local img2_arr = img2_arr[{{1,subset},{},{}}]
local disp_arr = disp_arr[{{1,subset},{},{}}]
local disp_max = disp_arr:max()
local img_w = img1_arr:size(3);

-- |define network|
local model = mcCnnFst(nbConvLayers, nbFeatureMap, kernel)
local hpatch = model.hpatch;
local _TR_NET_ = model:getMilNetBatch(img_w, disp_max) 
local _TE_NET_ = model:getTripletNet() 

if gpu_on then
  _TR_NET_:cuda()
  _TE_NET_:cuda()
end
_TR_PPARAM_, _TR_PGRAD_ = _TR_NET_:getParameters()
_TE_PPARAM_, _TE_PGRAD_ = _TE_NET_:getParameters()

-- |define datasets|
local trainSet = dl.unsup3EpiSet(img1_arr, img2_arr, hpatch, disp_max);
local testSet = dl.sup3PatchSet(img1_arr, img2_arr, disp_arr, hpatch);
testSet:shuffle();

-- |prepare test set|
_TE_INPUT_, _TE_TARGET_ = testSet:index(torch.range(1, test_set_size))
if gpu_on then
  _TE_TARGET_ = _TE_TARGET_:cuda()
  _TE_INPUT_[1] = _TE_INPUT_[1]:cuda();
  _TE_INPUT_[2] = _TE_INPUT_[2]:cuda();
  _TE_INPUT_[3] = _TE_INPUT_[3]:cuda();
end

-- |define criterion|
-- loss(x(+), x(-)) = max(0,  - x(+) + x(-)  + margin)
_CRITERION_ = nn.MarginRankingCriterion(margin);
if gpu_on then
  _CRITERION_:cuda()
end

-- |define  test function|
ftest = function()
    
    _TE_PPARAM_:copy(_TR_PPARAM_)
      
    local batch_size = _TE_INPUT_[1]:size(1);
    local patchRef, patchPos, patchNeg = unpack(_TE_INPUT_) 
    
    local test_acc = 0;
    for nsample = 1, batch_size do
    
      local sample_input = {{patchPos[{{nsample},{},{}}], patchNeg[{{nsample},{},{}}]}, patchRef[{{nsample},{},{}}]}
          
      -- forward pass
      local out = _TE_NET_:forward(sample_input)
      local pos_cos = torch.squeeze(out[1]:double())
      local neg_cos = torch.squeeze(out[2]:double())
      
      if pos_cos > neg_cos then
          test_acc = test_acc + 1;
      end
      
    end
    
    test_acc = (test_acc*100) / batch_size; 
        
    return test_acc    
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
    
    local sample_input = {{epiPos[{{nsample},{},{}}], epiNeg[{{nsample},{},{}}]}, epiRef[{{nsample},{},{}}]}
    local sample_target = _TR_TARGET_[{{nsample},{}}]    
        
    -- forward pass
    _TR_ERR_ = _TR_ERR_ + _CRITERION_:forward(_TR_NET_:forward(sample_input), sample_target)
  
    -- backword pass
    _TR_NET_:backward(sample_input, _CRITERION_:backward(_TR_NET_.output, sample_target))
    
  end
  _TR_ERR_ = _TR_ERR_ / batch_size
  _TR_PGRAD_:div(batch_size);
  
  return _TR_ERR_, _TR_PGRAD_      
end

-- |define optimization process|
config = {
   learningRate = 1e-1,
   momentum = 0.5,
   learningRateDecay = 1e-3
}

-- |define logger|
logger = optim.Logger('work/learning.log')
logger:setNames{'Training loss', 'Test accuracy'}
logger:style{'+-', '+-'}
    
-- |optimize network|   
local start_time = os.time()
for nepoch = 1, nb_epoch do

  nsample = 0;
  sample_err = {}
    
  for k, input, target in trainSet:sampleiter(batch_size, epoch_size) do
    
    _TR_INPUT_ = input
    _TR_TARGET_ = target  
      
    -- if gpu avaliable put batch on gpu
    if gpu_on then
      _TR_INPUT_[1] = _TR_INPUT_[1]:cuda()
      _TR_INPUT_[2] = _TR_INPUT_[2]:cuda()
      _TR_INPUT_[3] = _TR_INPUT_[3]:cuda()
      _TR_TARGET_ = _TR_TARGET_:cuda()
    end
        
    optim.sgd(feval, _TR_PPARAM_, config)    
    table.insert(sample_err, _TR_ERR_)
            
  end
  
  -- validation
  local test_acc = ftest()
  local train_err = torch.Tensor(sample_err):mean();
  
  -- save net
  local fNet = model:getFeatureNet()
  param = fNet:getParameters()
  param:copy(_TR_PPARAM_)
  torch.save('work/fnet.t7', fNet, 'ascii');
    
  
  local end_time = os.time()
  local time_diff = os.difftime(end_time,start_time);
  
  logger:add{train_err, test_acc}
  print(string.format("epoch %d, time = %f, train_err = %f, test_acc = %f", nepoch, time_diff, train_err, test_acc))
      
  if graph_on then
    logger:plot()
  end
      
end





