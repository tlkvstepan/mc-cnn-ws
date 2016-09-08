require 'nn'
mcCnnFst = dofile('CMcCnnFst.lua')
dofile('CAddMatrix.lua')
dofile('CMaxM.lua')

disp_max = 10
img_w = 30
hpatch = 4
fnet, hpatch = mcCnnFst.get(4, 64, 3)

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
local seqNegPos = nn.Sequential()
commutator1Net:add(seqRefPos)
commutator1Net:add(seqRefNeg)
commutator1Net:add(seqNegPos)
local selectorRefPos = nn.ConcatTable()
local selectorRefNeg = nn.ConcatTable()
local selectorNegPos = nn.ConcatTable()
seqRefPos:add(selectorRefPos)
seqRefNeg:add(selectorRefNeg)
seqNegPos:add(selectorNegPos)
selectorRefPos:add(nn.SelectTable(1))
selectorRefPos:add(nn.SelectTable(2))
selectorRefNeg:add(nn.SelectTable(1))
selectorRefNeg:add(nn.SelectTable(3))
selectorNegPos:add(nn.SelectTable(3))
selectorNegPos:add(nn.SelectTable(2))
seqRefPos:add(nn.MM(false, true))
seqRefNeg:add(nn.MM(false, true))
seqNegPos:add(nn.MM(false, true))

-- mask matrices
local mask = torch.ones(img_w-2*hpatch, img_w-2*hpatch)*2  
mask = torch.triu(torch.tril(mask,-1),-disp_max)
mask = mask - 2;
seqRefPos:add(nn.addMatrix(mask))
seqRefNeg:add(nn.addMatrix(mask))
seqNegPos:add(nn.addMatrix(mask))

-- ref-neg table we cut from the top, and take max along 2 dim
seqRefNeg:add(nn.Narrow(1, disp_max+1, img_w - 2*hpatch - disp_max))
seqRefNeg:add(nn.Max(2))

-- neg-pos table we cut from the right, transpose and take max along 2 dim
seqNegPos:add(nn.Narrow(2, 1, img_w - 2*hpatch - disp_max))
seqNegPos:add(nn.Transpose{1,2})
seqNegPos:add(nn.Max(2))

-- ref-pos is used twice, it is convenient to replicate it
seqRefPos:add(nn.Replicate(2))
seqRefPos:add(nn.SplitTable(1))

-- each copy of ref-pos we process differently
parRefPosPosRef = nn.ParallelTable()
seqRefPos:add(parRefPosPosRef)
subseqRefPos = nn.Sequential()
subseqPosRef = nn.Sequential()
parRefPosPosRef:add(subseqRefPos)
parRefPosPosRef:add(subseqPosRef)

-- first copy of ref-pos we cut from the top
subseqRefPos:add(nn.Narrow(1, disp_max+1, img_w - 2*hpatch - disp_max))

-- second copy of ref-pos we cut from the right and transpose
subseqPosRef:add(nn.Narrow(2, 1, img_w - 2*hpatch - disp_max))
subseqPosRef:add(nn.Transpose{1,2})

-- flatten tables hierarchy
-- after flattening, order is following ref-pos, pos-ref, ref-neg, pos-neg
Net:add(nn.FlattenTable())



---- make 4 streams: 
---- 2 stream for MIL cost : forward (ref-pos, ref-neg) and backward (pos-ref, neg-ref)
---- 2 streams for MAX2MAX cost : forward (ref-pos) and backward (pos-ref)
--local costCommutator = nn.ConcatTable()
--Net:add(costCommutator)
--local milForward = nn.Sequential()
--local milBackward = nn.Sequential()
--local max2maxForward = nn.Sequential()
--local max2maxBackward = nn.Sequential()
--costCommutator:add(milForward)
--costCommutator:add(milBackward)
--costCommutator:add(max2maxForward)
--costCommutator:add(max2maxBackward)
--local selectorMilForward = nn.ConcatTable()
--milForward:add(selectorMilForward) 
--local selectorMilBackward = nn.ConcatTable()
--milBackward:add(selectorMilBackward) 
--local selectorMax2maxForward = nn.ConcatTable()
--max2maxForward:add(selectorMax2maxForward) 
--local selectorMax2maxBackward = nn.ConcatTable()
--max2maxBackward:add(selectorMax2maxBackward) 
--selectorMilForward:add(nn.SelectTable(1)) 
--selectorMilForward:add(nn.SelectTable(3))
--selectorMilBackward:add(nn.SelectTable(2))
--selectorMilBackward:add(nn.SelectTable(4))
--selectorMax2maxForward:add(nn.SelectTable(1)) 
--selectorMax2maxBackward:add(nn.SelectTable(2))

------ compute forward mil
--local milForwardCompute = nn.ParallelTable()
--milForward:add(milForwardCompute)
--local seqRefPosMask = nn.Sequential()
--local seqRefNegMask = nn.Sequential()
--parRefPosRefNeg:add(seqRefPosMask)
--parRefPosRefNeg:add(seqRefNegMask)
--local mask = torch.ones(img_w-2*hpatch, img_w-2*hpatch)*2  
--mask = torch.triu(torch.tril(mask,-1),-disp_max)
--mask = mask - 2;
--seqRefNegMask:add(nn.addMatrix(mask))
--seqRefPosMask:add(nn.addMatrix(mask))
--seqRefPosMask:add(nn.Narrow(1,disp_max+1, img_w - 2*hpatch - disp_max))
--seqRefNegMask:add(nn.Narrow(1,disp_max+1, img_w - 2*hpatch - disp_max))
--seqRefNegMask:add(nn.Max(2))  
--seqRefPosMask:add(nn.Max(2))  
--seqRefNegMask:add(nn.Unsqueeze(2))  
--seqRefPosMask:add(nn.Unsqueeze(2)) 
--seqRefPosRefNeg:add(nn.JoinTable(2))  
  
---- compute backward output
--local parRefPosPosNeg = nn.ParallelTable()
--seqRefPosPosNeg:add(parRefPosPosNeg)
--local seqRefPosMask = nn.Sequential()
--local seqPosNegMask = nn.Sequential()
--parRefPosPosNeg:add(seqRefPosMask)
--parRefPosPosNeg:add(seqPosNegMask)
--seqPosNegMask:add(nn.Transpose({1,2}))
--local mask = torch.ones(img_w-2*hpatch, img_w-2*hpatch)*2  
--mask = torch.triu(torch.tril(mask,-1),-disp_max)
--mask = mask - 2;
--seqPosNegMask:add(nn.addMatrix(mask))
--seqRefPosMask:add(nn.addMatrix(mask))
--seqRefPosMask:add(nn.Narrow(2,1, img_w - 2*hpatch - disp_max))
--seqPosNegMask:add(nn.Narrow(2,1, img_w - 2*hpatch - disp_max))
--seqRefPosMask:add(nn.Transpose({1,2}))
--seqPosNegMask:add(nn.Transpose({1,2}))
--seqPosNegMask:add(nn.Max(2))  
--seqRefPosMask:add(nn.Max(2)) 
--seqPosNegMask:add(nn.Unsqueeze(2))  
--seqRefPosMask:add(nn.Unsqueeze(2)) 
--seqRefPosPosNeg:add(nn.JoinTable(2))

--Net:add(nn.JoinTable(1));
--Net:add(nn.SplitTable(2));



input = {torch.rand(1,2*hpatch+1,img_w), torch.rand(1,2*hpatch+1,img_w), torch.rand(1,2*hpatch+1,img_w)};

output = Net:forward(input);
--outcrit = criterion:forward(output,torch.Tensor{1})

print(output)