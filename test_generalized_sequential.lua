require 'nn'
mcCnnFst = dofile('CMcCnnFst.lua')
dofile('CAddMatrix.lua')
dim = 3;

disp_max = 10
img_w = 30
hpatch = 4
fNetRef, hpatch = mcCnnFst.get(4, 64, 3)



local tabH1 = img_w - 2*hpatch - disp_max;
local tabW1 = img_w - 2*hpatch;
local tabH2 = img_w - 2*hpatch - disp_max;
local tabW2 = img_w - 2*hpatch;

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
Net:add(nn.Clamp(-1,1))

-- make two streams : one with matrix and another with its transpose
splitter = nn.ConcatTable()
Net:add(splitter)
stream1 = nn.Sequential()
stream2 = nn.Sequential()
splitter:add(stream1)
splitter:add(stream2)

-- stream1 :
-- mask 
stream1:add(nn.Narrow(1,disp_max+1, img_w - 2*hpatch - disp_max))
-- perform softmax on rows
stream1:add(nn.SplitTable(1))
local par1 = nn.ParallelTable()
stream1:add(par1)
for i = 1, tabH1 do 
    local softmax = nn.Sequential()
    par1:add(softmax)
    softmax:add(nn.SoftMax()) 
    softmax:add(nn.Unsqueeze(1))
end
stream1:add(nn.JoinTable(1))
-- make entropys
stream1:add(nn.Replicate(2))
stream1:add(nn.SplitTable(1))
local par2 = nn.ParallelTable()
stream1:add(par2)
local seq1 = nn.Sequential()
par2:add(seq1)
seq1:add(nn.MulConstant(-1))
seq1:add(nn.View(1,tabH1*tabW1))
local seq2 = nn.Sequential()
par2:add(seq2)
seq2:add(nn.Log())
seq2:add(nn.View(tabH1*tabW1,1))
stream1:add(nn.MM(false, false))
stream1:add(nn.MulConstant(1/(tabH1)))

-- stream2 :
stream2:add(nn.Narrow(2,1, img_w - 2*hpatch - disp_max))
-- mask
stream2:add(nn.Transpose({1,2}))
-- perform softmax on rows
stream2:add(nn.SplitTable(1))
local par1 = nn.ParallelTable()
stream2:add(par1)
for i = 1, tabH2 do 
    local softmax = nn.Sequential()
    par1:add(softmax)
    softmax:add(nn.SoftMax()) 
    softmax:add(nn.Unsqueeze(1))
end
stream2:add(nn.JoinTable(1))
-- make entropys
stream2:add(nn.Replicate(2))
stream2:add(nn.SplitTable(1))
local par2 = nn.ParallelTable()
stream2:add(par2)
local seq1 = nn.Sequential()
par2:add(seq1)
seq1:add(nn.MulConstant(-1))
seq1:add(nn.View(1,tabH2*tabW2))
local seq2 = nn.Sequential()
par2:add(seq2)
seq2:add(nn.Log())
seq2:add(nn.View(tabH2*tabW2,1))
stream2:add(nn.MM(false, false))
stream2:add(nn.MulConstant(1/(tabH2)))
Net:add(nn.JoinTable(1))

-- add criterion (y should always be 1)
--criterion:add(nn.HingeEmbeddingCriterion(0.1))

input = {torch.rand(1,2*hpatch+1,img_w), torch.rand(1,2*hpatch+1,img_w)};
criterion = nn.HingeEmbeddingCriterion(1);
--input = torch.rand(tabH,tabW);

output = Net:forward(input);
outcrit = criterion:forward(output,torch.Tensor{{1},{1}})

print(output)