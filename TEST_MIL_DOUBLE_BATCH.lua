require 'torch'
require 'gnuplot'
require 'optim'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

dofile('DataLoader.lua');
dofile('CUnsup3EpiSet.lua');
dofile('CSup3PatchSet.lua');
dofile('CMcCnnFst.lua');
dofile('CAddMatrix.lua')

utils = dofile('utils.lua');

-- set randomseed to insure repeatability
math.randomseed(0); 

-- |parameteres|
-- learning
local test_set_size = 50000;       -- 50000 
local batch_size = 1           -- 512
local epoch_size = batch_size*1    -- 100
local nb_epoch = 1;            -- 10000
-- loss
local margin = 0.2;
-- network
nbFeatureMap = 64;
kernel = 3;
nbConvLayers = 5;
-- debug
local suffix = 'test_'
local gpu_on = false;
local debug = true;

if( gpu_on ) then
  require 'cunn'
end

fnet0 = torch.load('work/largeScale_fnet.t7', 'ascii')
local param0, grad0 = fnet0:getParameters()

-- |read data| from all KITTI
local img1_arr = torch.cat({torch.squeeze(utils.fromfile('data/KITTI12/x0.bin')),
    torch.squeeze(utils.fromfile('data/KITTI15/x0.bin'))},1);

local img2_arr = torch.cat({torch.squeeze(utils.fromfile('data/KITTI12/x1.bin')),
    torch.squeeze(utils.fromfile('data/KITTI15/x1.bin'))},1);

local disp_arr = torch.round(torch.squeeze(utils.fromfile('data/KITTI12/dispnoc.bin')));

local disp_max = disp_arr:max()
local img_w = img1_arr:size(3);

--print('Max disparity '.. disp_max.. '\n')
--print('Image width ' .. img_w ..'\n')


-- |define network|
_MODEL_ = mcCnnFst(nbConvLayers, nbFeatureMap, kernel)
local hpatch = _MODEL_.hpatch;
_TR_NET_ = _MODEL_:getMilNetDoubleBatch(img_w, disp_max) 
_TE_NET_ = _MODEL_:getTripletNet() 

if gpu_on then
  _TR_NET_:cuda()
  _TE_NET_:cuda()
end
_TR_PPARAM_, _TR_PGRAD_ = _TR_NET_:getParameters()
_TE_PPARAM_, _TE_PGRAD_ = _TE_NET_:getParameters()
_TR_PPARAM_:copy(param0);

-- |define datasets|
local trainSet = unsup3EpiSet(img1_arr, img2_arr, hpatch, disp_max);
local testSet = sup3PatchSet(img1_arr[{{1,194},{},{}}], img2_arr[{{1,194},{},{}}], disp_arr[{{1,194},{},{}}], hpatch);

-- |prepare test set|
_TE_INPUT_, target = testSet:index(torch.range(1, test_set_size))
_TE_TARGET_ = torch.ones(test_set_size)
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

    local sample_input = {epiRef[{{nsample},{},{}}],epiPos[{{nsample},{},{}}], epiNeg[{{nsample},{},{}}]}
    local sample_target = _TR_TARGET_[{{nsample},{}}]    

    -- forward pass
    _TR_ERR_ = _TR_ERR_ + _CRITERION_:forward(_TR_NET_:forward(sample_input), sample_target)

    -- backword pass
    _TR_NET_:backward(sample_input, _CRITERION_:backward(_TR_NET_.output, sample_target))

  end
  _TR_ERR_ = _TR_ERR_ / batch_size / 2
  _TR_PGRAD_:div(2*batch_size);

  return _TR_ERR_, _TR_PGRAD_      
end

-- |define logger|
if debug then
  logger = optim.Logger('work/'..suffix..'learning.log')
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
for nepoch = 1, nb_epoch do

  nsample = 0;
  sample_err = {}

  for k, input  in trainSet:sampleiter(batch_size, epoch_size) do

    _TR_INPUT_ = input
    _TR_TARGET_ =  torch.ones(batch_size, 2*(img_w - disp_max - 2*hpatch));  


    -- if gpu avaliable put batch on gpu
    if gpu_on then
      _TR_INPUT_[1] = _TR_INPUT_[1]:cuda()
      _TR_INPUT_[2] = _TR_INPUT_[2]:cuda()
      _TR_INPUT_[3] = _TR_INPUT_[3]:cuda()
      _TR_TARGET_ = _TR_TARGET_:cuda()
    end

    optim.adam(feval, _TR_PPARAM_, {})    
    table.insert(sample_err, _TR_ERR_)

  end

  -- validation
  local test_acc, test_acc_le2, test_acc_le5, err_index, disp_diff  = ftest()
  local train_err = torch.Tensor(sample_err):mean();

  local end_time = os.time()
  local time_diff = os.difftime(end_time,start_time);

  -- save debug info
  if debug then
    
    -- save errorneous test samples
    local fail_img = utils.vis_errors(_TE_INPUT_[1]:float():clone(), 
      _TE_INPUT_[2]:float():clone(), 
      _TE_INPUT_[3]:float():clone(), err_index, disp_diff)
    image.save('work/'..suffix..'failure.png',fail_img)

    -- save net
    local fNet = _MODEL_:getFeatureNet()
    local param = fNet:getParameters()
    param:copy(_TR_PPARAM_)
    torch.save('work/'..suffix..'fnet.t7', fNet, 'ascii');
    
    -- save log
    logger:add{train_err, test_acc, test_acc_le2, test_acc_le5}
    logger:plot()
    
    -- save distance matrices
    input = trainSet:index(torch.Tensor{290})
    _TR_NET_:forward({input[1]:cuda(),input[2]:cuda(),input[3]:cuda()} )
    refPos = _TR_NET_:get(2):get(1):get(2).output:clone():float();
    refPos = utils.mask(refPos,disp_max)
    refPos = utils.softmax(refPos)
    refPos = utils.scale2_01(refPos)
    image.save('work/'..suffix..'dist_ref_pos.png',refPos)
    
  end
  
  print(string.format("epoch %d, time = %f, train_err = %f, test_acc = %f", nepoch, time_diff, train_err, test_acc))
  collectgarbage()

end





