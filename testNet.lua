img_w = 20
disp_max = 3
hpatch = 5

nbFeatureMap = 64;
kernel = 3;
nbConvLayers = 5;


require 'torch'
require 'nn'
require 'cunn'
dofile('CAddMatrix.lua')


mask = 10*torch.ones(img_w-2*hpatch-disp_max,img_w-2*hpatch)  
mask = torch.triu(torch.tril(mask, disp_max))
mask = mask - 10;


dofile('CMcCnnFst.lua');

local model = mcCnnFst(nbConvLayers, nbFeatureMap, kernel)

local Net = nn.Sequential();
local parTab = nn.ParallelTable();
Net:add(parTab);

-- EPI NETs      
-- epipolar line we pass through same feature net and get 64 x 1 x epi_w
local epiSeq = nn.Sequential();
parTab:add(epiSeq)
local epiParTab = nn.ParallelTable();
epiSeq:add(epiParTab);
local epiNet1 = model:getFeatureNet();
local patchNet = epiNet1:clone('weight','bias', 'gradWeight','gradBias');
epiParTab:add(epiNet1);
local epiNet2 = epiNet1:clone('weight','bias', 'gradWeight','gradBias');
epiParTab:add(epiNet2);

-- squeeze 
local module = nn.Squeeze(2)
epiNet1:add(module)
local module = nn.Squeeze(2)
epiNet2:add(module)

-- transpose 
local module = nn.Transpose({1,2})
epiNet1:add(module)
local module = nn.Transpose({1,2})
epiNet2:add(module)

-- normalize
local module = nn.Normalize(2)
epiNet1:add(module)
local module = nn.Normalize(2)
epiNet2:add(module)

-- join
local  module = nn.JoinTable(1)
epiSeq:add(module);


local module = nn.Unsqueeze(1)
epiSeq:add(module)

-- PATCH NET
-- reference patch we pass through feature net and get 64 x 1 x 1 response
parTab:add(patchNet);

-- squeeze 
local module = nn.Squeeze(2)
patchNet:add(module)

-- transpose 
local module = nn.Transpose({1,2})
patchNet:add(module)

-- normalize
local module = nn.Normalize(2)
patchNet:add(module)

local module = nn.Unsqueeze(1)
patchNet:add(module)

-- multiply
local module = nn.MM(false, true)
Net:add(module)

-- split
module = nn.SplitTable(2)
Net:add(module)



patch_pos = torch.rand(1, 2*hpatch+1, 2*hpatch+1)
patch_neg = torch.rand(1, 2*hpatch+1, 2*hpatch+1) 
patch_ref = torch.rand(1,  2*hpatch+1, 2*hpatch+1)


Net:cuda()

out = Net:forward({{patch_pos:cuda(), patch_neg:cuda()}, patch_ref:cuda()})




print(out[1]:size())
print(out[2]:size())
print(out[3]:size())