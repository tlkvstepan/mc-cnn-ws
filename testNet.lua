require 'torch'
require 'gnuplot'
require 'nn'
require 'optim'
require 'cunn'

--require 'cutorch';

--------

local dl = require 'dataload'
dofile('CUnsup3EpiSet.lua');
dofile('CSup3PatchSet.lua');
dofile('CMcCnnFst.lua');
dofile('CAddMatrix.lua')

utils = dofile('utils.lua');


-- |read data|

hpatch = 5
local subset = 194
nbFeatureMap = 64;
kernel = 3;
nbConvLayers = 5;

local img1_arr = torch.squeeze(utils.fromfile('x0.bin')):double();
local img2_arr = torch.squeeze(utils.fromfile('x1.bin')):double();
local disp_arr = torch.round(torch.squeeze(utils.fromfile('dispnoc.bin'))):double();
local img1_arr = img1_arr[{{1,subset},{},{}}]
local img2_arr = img2_arr[{{1,subset},{},{}}]
local disp_arr = disp_arr[{{1,subset},{},{}}]
local disp_max = disp_arr:max()
local img_w = img1_arr:size(3);
local trainSet = dl.unsup3EpiSet(img1_arr, img2_arr, hpatch, disp_max);

local Net = nn.Sequential()

local model = mcCnnFst(nbConvLayers, nbFeatureMap, kernel)
-- pass 3 epipolar lines through feature net and normalize outputs
local parFeatureNet = nn.ParallelTable()
Net:add(parFeatureNet)
local fNetRef = model:getFeatureNet() 
fNetRef:add(nn.Squeeze(2))
fNetRef:add(nn.Transpose({1,2}))
fNetRef:add(nn.Normalize(2))
local fNetPos = fNetRef:clone('weight','bias', 'gradWeight','gradBias');
local fNetNeg = fNetRef:clone('weight','bias', 'gradWeight','gradBias');
parFeatureNet:add(fNetRef)
parFeatureNet:add(fNetPos)
parFeatureNet:add(fNetNeg)

-- compute 3 cross products: ref and pos, ref and neg, pos and neg
local commutator1Net = nn.ConcatTable()
Net:add(commutator1Net);
local seqRefPos = nn.Sequential()
local seqRefNeg = nn.Sequential()
commutator1Net:add(seqRefPos)
commutator1Net:add(seqRefNeg)
local selectorRefPos = nn.ConcatTable()
local selectorRefNeg = nn.ConcatTable()
seqRefPos:add(selectorRefPos)
seqRefNeg:add(selectorRefNeg)
selectorRefPos:add(nn.SelectTable(1))
selectorRefPos:add(nn.SelectTable(2))
selectorRefNeg:add(nn.SelectTable(1))
selectorRefNeg:add(nn.SelectTable(3))
seqRefPos:add(nn.MM(false, true))
seqRefNeg:add(nn.MM(false, true))

-- make 2 streams: forward and backward cost
--local commutator2Net = nn.ConcatTable()
--Net:add(commutator2Net)
--local seqRefPosRefNeg = nn.Sequential()
--commutator2Net:add(seqRefPosRefNeg)
--local selectorRefPosRefNeg = nn.ConcatTable()
--seqRefPosRefNeg:add(selectorRefPosRefNeg) 
--selectorRefPosRefNeg:add(nn.SelectTable(1))
--selectorRefPosRefNeg:add(nn.SelectTable(2))

-- compute forward output
local parRefPosRefNeg = nn.ParallelTable()
Net:add(parRefPosRefNeg)
local seqRefPosMask = nn.Sequential()
local seqRefNegMask = nn.Sequential()
parRefPosRefNeg:add(seqRefPosMask)
parRefPosRefNeg:add(seqRefNegMask)
local mask = 2*torch.ones(img_w-2*hpatch, img_w-2*hpatch)  
mask = torch.triu(torch.tril(mask,-1),-disp_max)
mask = mask - 2;
seqRefNegMask:add(nn.addMatrix(mask))
seqRefPosMask:add(nn.addMatrix(mask))
seqRefPosMask:add(nn.Narrow(1,disp_max+1, img_w - 2*hpatch - disp_max))
seqRefNegMask:add(nn.Narrow(1,disp_max+1, img_w - 2*hpatch - disp_max))
seqRefNegMask:add(nn.Max(2))  
seqRefPosMask:add(nn.Max(2))  

--seqRefPosRefNeg:add(nn.FlattenTable())  


--seqRefPosPosNegCost = nn.Sequential()
--parCostCompute:add(seqRefPosRefNegCost)
--parCostCompute:add(seqRefPosPosNegCost)


epi_ref = torch.rand(1,11,img_w):cuda();
epi_pos = torch.rand(1,11,img_w):cuda();
epi_neg = torch.rand(1,11,img_w):cuda(); 

--inputs[1] = inputs[1]:cuda()
--inputs[2] = inputs[2]:cuda()
--inputs[3] = inputs[3]:cuda()

Net:cuda()
out = Net:forward({epi_ref,epi_pos,epi_neg})


x=out