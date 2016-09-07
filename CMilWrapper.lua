local milWrapper = {}

function milWrapper.getMilNetDoubleBatch(img_w, disp_max, hpatch, fnet)
  
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
seqRefNegMask:add(nn.Unsqueeze(2))  
seqRefPosMask:add(nn.Unsqueeze(2)) 
seqRefPosRefNeg:add(nn.JoinTable(2))  
  
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
seqPosNegMask:add(nn.Unsqueeze(2))  
seqRefPosMask:add(nn.Unsqueeze(2)) 
seqRefPosPosNeg:add(nn.JoinTable(2))

Net:add(nn.JoinTable(1));
Net:add(nn.SplitTable(2));

return Net

end


function milWrapper.getMilNetBatch(img_w, disp_max, hpatch, fnet)
  
local fNetRef = fnet:clone();  
local Net = nn.Sequential()

-- pass 3 epipolar lines through feature net and normalize outputs
local parFeatureNet = nn.ParallelTable()
Net:add(parFeatureNet)
local fNetRef = self:getFeatureNet() 
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

-- compute forward output
local parRefPosRefNeg = nn.ParallelTable()
Net:add(parRefPosRefNeg)
local seqRefPosMask = nn.Sequential()
local seqRefNegMask = nn.Sequential()
parRefPosRefNeg:add(seqRefPosMask)
parRefPosRefNeg:add(seqRefNegMask)
local mask = 2*torch.ones(img_w-2*self.hpatch, img_w-2*self.hpatch)  
mask = torch.triu(torch.tril(mask,-1),-disp_max)
mask = mask - 2;
seqRefNegMask:add(nn.addMatrix(mask))
seqRefPosMask:add(nn.addMatrix(mask))
seqRefPosMask:add(nn.Narrow(1,disp_max+1, img_w - 2*self.hpatch - disp_max))
seqRefNegMask:add(nn.Narrow(1,disp_max+1, img_w - 2*self.hpatch - disp_max))
seqRefNegMask:add(nn.Max(2))  
seqRefPosMask:add(nn.Max(2))  

return Net

end

function milWrapper.getEntropyNetDoubleBatch(img_w, disp_max, hpatch, fnet)  

  local fNetRef = fnet:clone(); 

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
--local mask = torch.ones(img_w-2*hpatch, img_w-2*hpatch)*2  
--mask = torch.triu(torch.tril(mask,-1),-disp_max)
--mask = mask - 2;
--Net:add(nn.addMatrix(mask))
--Net:add(nn.Clamp(-1,1))

-- make two streams : one with matrix and another with its transpose
--splitter = nn.ConcatTable()
--Net:add(splitter)
--stream1 = nn.Sequential()
--stream2 = nn.Sequential()
--splitter:add(stream1)
--splitter:add(stream2)

-- stream1 :
-- mask 
Net:add(nn.Narrow(1,disp_max+1, img_w - 2*hpatch - disp_max))
-- perform softmax on rows
Net:add(nn.SplitTable(1))
local par1 = nn.ParallelTable()
Net:add(par1)
for i = 1, tabH1 do 
    local softmax = nn.Sequential()
    -- we want to compute entropy only for valid values
    par1:add(softmax)
    softmax:add(nn.Narrow(1,i,disp_max))
    softmax:add(nn.SoftMax()) 
    softmax:add(nn.Unsqueeze(1))
end
Net:add(nn.JoinTable(1))
---- make entropys
Net:add(nn.Replicate(2))
Net:add(nn.SplitTable(1))
local par2 = nn.ParallelTable()
Net:add(par2)
local seq1 = nn.Sequential()
par2:add(seq1)
seq1:add(nn.MulConstant(-1))
seq1:add(nn.View(1,tabH1*disp_max))
local seq2 = nn.Sequential()
par2:add(seq2)
seq2:add(nn.Log())
seq2:add(nn.View(tabH1*disp_max,1))
Net:add(nn.MM(false, false))
Net:add(nn.MulConstant(1/(tabH1)))

-- stream2 :
--stream2:add(nn.Narrow(2,1, img_w - 2*hpatch - disp_max))
---- mask
--stream2:add(nn.Transpose({1,2}))
---- perform softmax on rows
--stream2:add(nn.SplitTable(1))
--local par1 = nn.ParallelTable()
--stream2:add(par1)
--for i = 1, tabH2 do 
--    local softmax = nn.Sequential()
--    par1:add(softmax)
--    softmax:add(nn.SoftMax()) 
--    softmax:add(nn.Unsqueeze(1))
--end
--stream2:add(nn.JoinTable(1))
---- make entropys
--stream2:add(nn.Replicate(2))
--stream2:add(nn.SplitTable(1))
--local par2 = nn.ParallelTable()
--stream2:add(par2)
--local seq1 = nn.Sequential()
--par2:add(seq1)
--seq1:add(nn.MulConstant(-1))
--seq1:add(nn.View(1,tabH2*tabW2))
--local seq2 = nn.Sequential()
--par2:add(seq2)
--seq2:add(nn.Log())
--seq2:add(nn.View(tabH2*tabW2,1))
--stream2:add(nn.MM(false, false))
--stream2:add(nn.MulConstant(1/(tabH2)))
--Net:add(nn.JoinTable(1))

-- add criterion (y should always be 1)
--criterion:add(nn.HingeEmbeddingCriterion(0.1))
  
  return Net
end

-- Function creates triplet network by combining 3 feature nets
-- Input of network is table of 3 of 1 x patch_height x patch_width 
-- (order is important Patch, Positive, Negative)
-- Output of network is table of 2 numbers
function milWrapper.getTripletNet(fnet)
      
  epiNet1 = fnet:clone();
  
  local Net = nn.Sequential();
  local parTab = nn.ParallelTable();
  Net:add(parTab);

  -- EPI NETs      
  -- epipolar line we pass through same feature net and get 64 x 1 x epi_w
  local epiSeq = nn.Sequential();
  parTab:add(epiSeq)
  local epiParTab = nn.ParallelTable();
  epiSeq:add(epiParTab);
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
  local module = nn.SplitTable(2)
  Net:add(module)
  
  return Net
  
end
 
return milWrapper
