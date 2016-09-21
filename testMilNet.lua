require 'nn'
require 'cunn'
baseNet = dofile('CBaseNet.lua')
dofile('CAddMatrix.lua')
require 'libdynprog'
dofile('CDprog.lua')
dofile('CContrastDprog.lua')


disp_max = 200
img_w = 1000
hpatch = 4
max_order = 3
fnet, hpatch = baseNet.get(4, 64, 3)
dist_min = 2
local fNetRef = fnet:clone();
 
local fNetRef = fnet:clone();
 
local Net = nn.Sequential()

-- pass 3 epipolar lines through feature net and normalize outputs
local parFeatureNet = nn.ParallelTable()
Net:add(parFeatureNet)
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
local seqPosNeg = nn.Sequential()
commutator1Net:add(seqRefPos)
commutator1Net:add(seqRefNeg)
commutator1Net:add(seqPosNeg)
local selectorRefPos = nn.ConcatTable()
local selectorRefNeg = nn.ConcatTable()
local selectorPosNeg = nn.ConcatTable()
seqRefPos:add(selectorRefPos)
seqRefNeg:add(selectorRefNeg)
seqPosNeg:add(selectorPosNeg)
selectorRefPos:add(nn.SelectTable(1))
selectorRefPos:add(nn.SelectTable(2))
selectorRefNeg:add(nn.SelectTable(1))
selectorRefNeg:add(nn.SelectTable(3))
selectorPosNeg:add(nn.SelectTable(2))
selectorPosNeg:add(nn.SelectTable(3))
seqRefPos:add(nn.MM(false, true))
seqRefNeg:add(nn.MM(false, true))
seqPosNeg:add(nn.MM(false, true))

-- make 2 streams: forward and backward cost
local commutator2Net = nn.ConcatTable()
Net:add(commutator2Net)
local seqRefPosRefNeg = nn.Sequential()
local seqRefPosPosNeg = nn.Sequential()
commutator2Net:add(seqRefPosRefNeg)
commutator2Net:add(seqRefPosPosNeg)
local selectorRefPosRefNeg = nn.ConcatTable()
seqRefPosRefNeg:add(selectorRefPosRefNeg) 
local selectorRefPosPosNeg = nn.ConcatTable()
seqRefPosPosNeg:add(selectorRefPosPosNeg) 
selectorRefPosRefNeg:add(nn.SelectTable(1))
selectorRefPosRefNeg:add(nn.SelectTable(2))
selectorRefPosPosNeg:add(nn.SelectTable(1))
selectorRefPosPosNeg:add(nn.SelectTable(3))

-- compute forward output
local parRefPosRefNeg = nn.ParallelTable()
seqRefPosRefNeg:add(parRefPosRefNeg)
local seqRefPosMask = nn.Sequential()
local seqRefNegMask = nn.Sequential()
parRefPosRefNeg:add(seqRefPosMask)
parRefPosRefNeg:add(seqRefNegMask)
local mask = torch.ones(img_w-2*hpatch, img_w-2*hpatch)*2  
mask = torch.triu(torch.tril(mask,-1),-disp_max)
mask = mask - 2;
seqRefNegMask:add(nn.addMatrix(mask))
seqRefPosMask:add(nn.addMatrix(mask))
seqRefPosMask:add(nn.Narrow(1,disp_max+1, img_w - 2*hpatch - disp_max))
seqRefNegMask:add(nn.Narrow(1,disp_max+1, img_w - 2*hpatch - disp_max))
seqRefNegMask:add(nn.Max(2))  
seqRefPosMask:add(nn.Max(2))  
--seqRefNegMask:add(nn.Unsqueeze(2))  
--seqRefPosMask:add(nn.Unsqueeze(2)) 
--seqRefPosRefNeg:add(nn.JoinTable(2))  
  
-- compute backward output
local parRefPosPosNeg = nn.ParallelTable()
seqRefPosPosNeg:add(parRefPosPosNeg)
local seqRefPosMask = nn.Sequential()
local seqPosNegMask = nn.Sequential()
parRefPosPosNeg:add(seqRefPosMask)
parRefPosPosNeg:add(seqPosNegMask)
seqPosNegMask:add(nn.Transpose({1,2}))
local mask = torch.ones(img_w-2*hpatch, img_w-2*hpatch)*2  
mask = torch.triu(torch.tril(mask,-1),-disp_max)
mask = mask - 2;
seqPosNegMask:add(nn.addMatrix(mask))
seqRefPosMask:add(nn.addMatrix(mask))
seqRefPosMask:add(nn.Narrow(2,1, img_w - 2*hpatch - disp_max))
seqPosNegMask:add(nn.Narrow(2,1, img_w - 2*hpatch - disp_max))
seqRefPosMask:add(nn.Transpose({1,2}))
seqPosNegMask:add(nn.Transpose({1,2}))
seqPosNegMask:add(nn.Max(2))  
seqRefPosMask:add(nn.Max(2)) 
--seqPosNegMask:add(nn.Unsqueeze(2))  
--seqRefPosMask:add(nn.Unsqueeze(2)) 
--seqRefPosPosNeg:add(nn.JoinTable(2))

-- cpu
start_cpu = os.time()
input_cpu = {torch.rand(1,2*hpatch+1,img_w), torch.rand(1,2*hpatch+1,img_w), torch.rand(1,2*hpatch+1,img_w)};
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

