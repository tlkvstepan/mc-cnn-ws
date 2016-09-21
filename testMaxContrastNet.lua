require 'nn'
mcCnnFst = dofile('CBaseNet.lua')
dofile('CAddMatrix.lua')
dofile('CContrastMax.lua')

disp_max = 228
img_w = 1242
hpatch = 4
dist_min = 2
fNetRef, hpatch = mcCnnFst.get(4, 64, 3)


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

-- make two streams : one with matrix and another with its transpose
splitter = nn.ConcatTable()
Net:add(splitter)
stream1 = nn.Sequential()
stream2 = nn.Sequential()
splitter:add(stream1)
splitter:add(stream2)

-- stream1 :
stream1:add(nn.Narrow(1, disp_max+1, img_w - 2*hpatch - disp_max))
stream1:add(nn.contrastMax(dist_min))
stream1:add(nn.SplitTable(2))

-- stream2 :
stream2:add(nn.Narrow(2,1, img_w - 2*hpatch - disp_max))
stream2:add(nn.Transpose({1,2}))
stream2:add(nn.contrastMax(dist_min))
stream2:add(nn.SplitTable(2))

--Net:add(nn.JoinTable(1))

-- add criterion (y should always be 1)
--criterion:add(nn.HingeEmbeddingCriterion(0.1))

input = {torch.rand(1,2*hpatch+1,img_w), torch.rand(1,2*hpatch+1,img_w),torch.rand(1,2*hpatch+1,img_w) };
criterion = nn.HingeEmbeddingCriterion(1);

output = Net:forward(input);
outcrit = criterion:forward(output,torch.Tensor{1})

print(output)