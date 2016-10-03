require 'nn'
require 'gnuplot'
--require 'cunn'
require 'libdprog'
mcCnnFst = dofile('CBaseNet.lua')
dofile('CAddMatrix.lua')
dofile('CMilDprog.lua')


dofile('DataLoader.lua');                 -- Parent class for dataloaders
dofile('CUnsup3EpiSet.lua');              -- Unsupervised training set loader
dofile('CSup2EpiSet.lua');          -- Supervised validation set loader

baseNet = dofile('CBaseNet.lua');         -- Function that makes base net
netWrapper = dofile('CNetWrapper.lua')    -- Function that "wrap" base net into training net
testFun = dofile('CTestUtils.lua');         -- Function that performs test on validation set

utils = dofile('utils.lua');              -- Utils for loading and visualization


-- |read trainng data| (KITTI)
local img1_arr = torch.squeeze(utils.fromfile('data/KITTI12/x0.bin'));
local img2_arr = torch.squeeze(utils.fromfile('data/KITTI12/x1.bin'));
local disp_arr = torch.round(torch.squeeze(utils.fromfile('data/KITTI12/dispnoc.bin')));

local disp_max = disp_arr:max()
local img_w = img1_arr:size(3);
hpatch = 4


local unsupSet = unsup3EpiSet(img1_arr, img2_arr, hpatch, disp_max);

-- get validation set from supervised set
input, target = unsupSet:index(torch.Tensor{22})

fNetRef, hpatch = mcCnnFst.get(4, 64, 3)

local Net = nn.Sequential()

-- pass 3 epipolar lines through feature net and normalize outputs
local fNets = nn.ParallelTable()
Net:add(fNets)  -- parallel feature nets
fNetRef:add(nn.Squeeze(2))
fNetRef:add(nn.Transpose({1,2}))
fNetRef:add(nn.Normalize(2))
local fNetPos = fNetRef:clone('weight','bias', 'gradWeight','gradBias');
local fNetNeg = fNetRef:clone('weight','bias', 'gradWeight','gradBias');
fNets:add(fNetRef)
fNets:add(fNetPos)
fNets:add(fNetNeg)

-- compute 3 cross products: ref and pos, ref and neg, pos and neg
local fNets2dNetCom = nn.ConcatTable()
Net:add(fNets2dNetCom); -- feature net to distance net commutator
local dNetRefPos_ = nn.Sequential()
local dNetRefNeg = nn.Sequential()
local dNetNegPos = nn.Sequential()
fNets2dNetCom:add(dNetRefPos_)
fNets2dNetCom:add(dNetRefNeg)
fNets2dNetCom:add(dNetNegPos)
local dNetRefPosSel = nn.ConcatTable()  -- input selectors for each distance net
local dNetRefNegSel = nn.ConcatTable()
local dNetNegPosSel = nn.ConcatTable()
dNetRefPos_:add(dNetRefPosSel)
dNetRefNeg:add(dNetRefNegSel)
dNetNegPos:add(dNetNegPosSel)
dNetRefPosSel:add(nn.SelectTable(1))
dNetRefPosSel:add(nn.SelectTable(2))
dNetRefNegSel:add(nn.SelectTable(1))
dNetRefNegSel:add(nn.SelectTable(3))
dNetNegPosSel:add(nn.SelectTable(3))
dNetNegPosSel:add(nn.SelectTable(2))
dNetRefPos_:add(nn.MM(false, true))
dNetRefNeg:add(nn.MM(false, true))
dNetNegPos:add(nn.MM(false, true))

-- mask distance matrices to leave only valid disparities
-- Basically we substract 2 from elements that should be ignored.
-- In this way we make these elements much smaller than other elements.
local mask = torch.ones(img_w-2*hpatch, img_w-2*hpatch)*2  
mask = torch.triu(torch.tril(mask,-1),-disp_max)
mask = mask - 2; 
dNetRefPos_:add(nn.addMatrix(mask))
dNetRefNeg:add(nn.addMatrix(mask))
dNetNegPos:add(nn.addMatrix(mask))

-- clamp (-1, 1)
dNetRefPos_:add(nn.Clamp(-1,1))
dNetRefNeg:add(nn.Clamp(-1,1))
dNetNegPos:add(nn.Clamp(-1,1))

-- convert range to (0 1)
dNetRefPos_:add(nn.AddConstant(1))
dNetRefPos_:add(nn.MulConstant(0.5))
dNetRefNeg:add(nn.AddConstant(1))
dNetRefNeg:add(nn.MulConstant(0.5))
dNetNegPos:add(nn.AddConstant(1))
dNetNegPos:add(nn.MulConstant(0.5))

-- milDprog
Net:add(nn.milDprog())

-- split each table into two
local splitNet = nn.ParallelTable()
Net:add(splitNet)
splitNet:add(nn.SplitTable(2))
splitNet:add(nn.SplitTable(2))
---- make 2 copies of refPos distance matrix, since we use will use it twice 
---- as ref-pos and pos-ref
--dNetRefPos_:add(nn.Replicate(2))
--dNetRefPos_:add(nn.SplitTable(1))
--local dNetRefPosSpl_ = nn.ParallelTable() -- splitter for ref-pos distance matrix
--dNetRefPos_:add(dNetRefPosSpl_)
--local dNetRefPos = nn.Sequential()
--local dNetPosRef = nn.Sequential()
--dNetRefPosSpl_:add(dNetRefPos)
--dNetRefPosSpl_:add(dNetPosRef)

---- now cut parts of distance matrices, that correspond to edges of the image
---- since on the edges of the image we might not have correct matches
---- ref-neg table we cut from the top, and take max along 2 dim
--dNetRefNeg:add(nn.Narrow(1, disp_max+1, img_w - 2*hpatch - disp_max))
---- neg-pos table we cut from the right, transpose and take max along 2 dim
--dNetNegPos:add(nn.Narrow(2, 1, img_w - 2*hpatch - disp_max))
--dNetNegPos:add(nn.Transpose{1,2})
---- ref-pos we cut from the top
--dNetRefPos:add(nn.Narrow(1, disp_max+1, img_w - 2*hpatch - disp_max))
---- second copy of ref-pos we cut from the right and transpose to obtain pos-ref
--dNetPosRef:add(nn.Narrow(2, 1, img_w - 2*hpatch - disp_max))
--dNetPosRef:add(nn.Transpose{1,2})

---- compute row-wise maximums
--dNetRefPos:add(nn.Dprog())
--dNetRefNeg:add(nn.Dprog())
--dNetPosRef:add(nn.Dprog())
--dNetNegPos:add(nn.Dprog())

---- flatten tables hierarchy
---- after flattening, order is following 
---- ref-pos-max, pos-ref-max, ref-neg-max, pos-neg-max
--Net:add(nn.FlattenTable())

---- make 2 output tables of tables
--local dNet2CostCom = nn.ConcatTable()
--Net:add(dNet2CostCom); -- feature net to distance net commutator
--local milFwd = nn.Sequential()
--local milBwd = nn.Sequential()
--dNet2CostCom:add(milFwd)
--dNet2CostCom:add(milBwd)
--local milFwdSel = nn.ConcatTable()  -- input selectors for cost
--local milBwdSel = nn.ConcatTable()
--milFwd:add(milFwdSel)
--milBwd:add(milBwdSel)
--milFwdSel:add(nn.SelectTable(1))  -- ref-pos-max
--milFwdSel:add(nn.SelectTable(3))  -- ref-neg-max
--milBwdSel:add(nn.SelectTable(2))  -- pos-ref-max
--milBwdSel:add(nn.SelectTable(4))  -- pos-neg-max

--local milFwdCst = nn.MarginRankingCriterion(loss_margin);
--local milBwdCst = nn.MarginRankingCriterion(loss_margin);
--local criterion = nn.ParallelCriterion(true):add(milFwdCst,0.5):add(milBwdCst,0.5)

--return Net, criterion

-- cpu
start_cpu = os.time()
input_cpu = input;
output_cpu = Net:forward(input_cpu);
end_cpu = os.time()
print(os.difftime(end_cpu, start_cpu))


-- cuda
input = {torch.rand(1,2*hpatch+1,img_w):cuda(), torch.rand(1,2*hpatch+1,img_w):cuda(), torch.rand(1,2*hpatch+1,img_w):cuda()};
Net:cuda()
start_gpu = os.time()
output_gpu = Net:forward(input)
end_gpu = os.time()
print(os.difftime(end_gpu, start_gpu))

criOut = parCri:forward(netOut, targ)
outGradCri = parCri:backward(netOut, targ)
  
--Net:backward(netIn, parCri:backward(criIn, criTarg))

--outcrit = criterion:forward(output,torch.Tensor{1})

