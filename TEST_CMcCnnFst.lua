require 'torch'
require 'nn'

local dl = require 'dataload'
dofile('CUnsup3EpiSet.lua');

dofile('CMcCnnFst.lua');
utils = dofile('UTILS.lua');

-- |define params|
-- learning
local epoch_size = 256*10
local batch_size = 256
local nb_epoch = 10
-- net
nbFeatureMap = 64;
kernel = 3;
nbConvLayers = 5;
hpatch = 5

-- |read data|
local img1_arr = torch.squeeze(utils.fromfile('x0.bin')):double();
local img2_arr = torch.squeeze(utils.fromfile('x1.bin')):double();
local disp_arr = torch.round(torch.squeeze(utils.fromfile('dispnoc.bin'))):double();
local img1_arr = img1_arr[{{1,subset_size},{},{}}]
local img2_arr = img2_arr[{{1,subset_size},{},{}}]
local disp_arr = disp_arr[{{1,subset_size},{},{}}]
local disp_max = disp_arr:max()
local img_w = img1_arr:size(3);
local set = dl.unsup3EpiSet(img1_arr, img2_arr, hpatch, disp_max);

local model = mcCnnFst(nbConvLayers, nbFeatureMap, kernel)
local net = model:getMilNetBatch(img_w, disp_max)
local pParam, pGrad = net:getParameters()

margin = 0.2;
criterion = nn.MarginRankingCriterion(margin);

for k, batch_inputs, batch_targets in set:sampleiter(1, epoch_size) do
  pGrad:zero()
  local epiRef, epiPos, epiNeg = unpack(batch_inputs)
  local input = {{epiPos, epiNeg}, epiRef}
  err1 = criterion:forward(net:forward(input), batch_targets);    
  
   local input = {{epiNeg,epiPos}, epiRef}
  err2 = criterion:forward(net:forward(input), batch_targets);    
  --net:backward(input, criterion:backward(net.output, batch_targets));

end