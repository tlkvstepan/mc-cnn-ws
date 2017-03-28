
require 'nn'

require 'libdprog'                        -- C++ module for dynamic programming
mcCnnFst = dofile('CBaseNet.lua')
dofile('CAddMatrix.lua')
dofile('CContrastDprog.lua')

disp_max = 228
img_w = 1242
hpatch = 4
fNetRef, hpatch = mcCnnFst.get(4, 64, 3)
dist_min = 2


local Net = nn.Sequential()
-- pass ref and pos epipolar lines through feature net and normalize outputs
local parFeatureNet = nn.ParallelTable()
Net:add(parFeatureNet)
fNetRef:add(nn.Squeeze(2))
fNetRef:add(nn.Transpose({1,2}))
fNetRef:add(nn.Normalize(2))
local fNetPos = fNetRef:clone('weight','bias', 'gradWeight','gradBias');
parFeatureNet:add(fNetRef)
parFeatureNet:add(fNetPos)

-- compute cross products ref and pos
Net:add(nn.MM(false, true))

-- mask wrong disparities
local mask = torch.ones(img_w-2*hpatch, img_w-2*hpatch)*2  
mask = torch.triu(torch.tril(mask,-1),-disp_max)
mask = mask - 2;
Net:add(nn.addMatrix(mask))

Net:add(nn.contrastDprog(dist_min))
Net:add(nn.SplitTable(2))

local NetCom = nn.ConcatTable()
Net:add(NetCom); -- feature net to distance net commutator
local NetFCost = nn.Sequential()
local NetBCost = nn.Sequential()
NetCom:add(NetFCost)
NetCom:add(NetBCost)
local NetFCostSel = nn.ConcatTable()  -- input selectors for each distance net
local NetBCostSel = nn.ConcatTable()
NetFCost:add(NetFCostSel)
NetBCost:add(NetBCostSel)
NetFCostSel:add(nn.SelectTable(1))
NetFCostSel:add(nn.SelectTable(2))
NetBCostSel:add(nn.SelectTable(3))
NetBCostSel:add(nn.SelectTable(4))


input = {torch.rand(1,2*hpatch+1,img_w), torch.rand(1,2*hpatch+1,img_w)};

output = Net:forward(input);

print(output)