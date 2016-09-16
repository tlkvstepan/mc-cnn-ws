local milWrapper = {}

function milWrapper.getMilMaxDoubleBatch(img_w, disp_max, hpatch, max_order, max_r, fnet)

local fNetRef = fnet:clone();
 
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

-- make 2 copies of refPos distance matrix, since we use will use it twice 
-- as ref-pos and pos-ref
dNetRefPos_:add(nn.Replicate(2))
dNetRefPos_:add(nn.SplitTable(1))
local dNetRefPosSpl_ = nn.ParallelTable() -- splitter for ref-pos distance matrix
dNetRefPos_:add(dNetRefPosSpl_)
local dNetRefPos = nn.Sequential()
local dNetPosRef = nn.Sequential()
dNetRefPosSpl_:add(dNetRefPos)
dNetRefPosSpl_:add(dNetPosRef)

-- now cut parts of distance matrices, that correspond to edges of the image
-- since on the edges of the image we might not have correct matches
-- ref-neg table we cut from the top, and take max along 2 dim
dNetRefNeg:add(nn.Narrow(1, disp_max+1, img_w - 2*hpatch - disp_max))
-- neg-pos table we cut from the right, transpose and take max along 2 dim
dNetNegPos:add(nn.Narrow(2, 1, img_w - 2*hpatch - disp_max))
dNetNegPos:add(nn.Transpose{1,2})
-- ref-pos we cut from the top
dNetRefPos:add(nn.Narrow(1, disp_max+1, img_w - 2*hpatch - disp_max))
-- second copy of ref-pos we cut from the right and transpose to obtain pos-ref
dNetPosRef:add(nn.Narrow(2, 1, img_w - 2*hpatch - disp_max))
dNetPosRef:add(nn.Transpose{1,2})

-- pos-ref and ref-pos matrices we will use twice to compute rowwise max and 
-- rowwise second max, therefore lets split them 
dNetRefPos:add(nn.Replicate(2))
dNetRefPos:add(nn.SplitTable(1))
local dNetRefPosSpl = nn.ParallelTable() -- splitter for ref-pos distance matrix
dNetRefPos:add(dNetRefPosSpl)
local dNetRefPosMax = nn.Sequential()
local dNetRefPosMaxM = nn.Sequential()
dNetRefPosSpl:add(dNetRefPosMax)
dNetRefPosSpl:add(dNetRefPosMaxM)

dNetPosRef:add(nn.Replicate(2))
dNetPosRef:add(nn.SplitTable(1))
local dNetPosRefSpl = nn.ParallelTable() -- splitter for ref-pos distance matrix
dNetPosRef:add(dNetPosRefSpl)
local dNetPosRefMax = nn.Sequential()
local dNetPosRefMaxM = nn.Sequential()
dNetPosRefSpl:add(dNetPosRefMax)
dNetPosRefSpl:add(dNetPosRefMaxM)

-- now compute max for: posRefMax, refPosMax, refNeg, posNeg
-- and Mth max for: posRefMaxM and refPosMaxM
dNetPosRefMax:add(nn.Max(2))
dNetRefPosMax:add(nn.Max(2))
dNetRefNeg:add(nn.Max(2))
dNetNegPos:add(nn.Max(2))
dNetPosRefMaxM:add(nn.MaxM(2, max_order, max_r))
dNetRefPosMaxM:add(nn.MaxM(2, max_order, max_r))


-- flatten tables hierarchy
-- after flattening, order is following 
-- ref-pos-max, ref-pos-maxm, pos-ref-max, pos-ref-maxm, ref-neg-max, pos-neg-max
Net:add(nn.FlattenTable())

-- make 4 output tables of tables
local dNet2CostCom = nn.ConcatTable()
Net:add(dNet2CostCom); -- feature net to distance net commutator
local milFwd = nn.Sequential()
local milBwd = nn.Sequential()
local maxFwd = nn.Sequential()
local maxBwd = nn.Sequential()
dNet2CostCom:add(milFwd)
dNet2CostCom:add(milBwd)
dNet2CostCom:add(maxFwd)
dNet2CostCom:add(maxBwd)
local milFwdSel = nn.ConcatTable()  -- input selectors for cost
local milBwdSel = nn.ConcatTable()
local maxFwdSel = nn.ConcatTable()  -- input selectors for each distance net
local maxBwdSel = nn.ConcatTable()
milFwd:add(milFwdSel)
milBwd:add(milBwdSel)
maxFwd:add(maxFwdSel)
maxBwd:add(maxBwdSel)
milFwdSel:add(nn.SelectTable(1))  -- ref-pos-max
milFwdSel:add(nn.SelectTable(5))  -- ref-neg-max
milBwdSel:add(nn.SelectTable(3))  -- pos-ref-max
milBwdSel:add(nn.SelectTable(6))  -- ref-neg-max
maxFwdSel:add(nn.SelectTable(1))  -- ref-pos-max
maxFwdSel:add(nn.SelectTable(2))  -- ref-pos-maxm
maxBwdSel:add(nn.SelectTable(3))  -- ref-pos-max
maxBwdSel:add(nn.SelectTable(4))  -- ref-pos-maxm

return Net;
end

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

function milWrapper.getMaxNetDoubleBatch(img_w, disp_max, hpatch, dist_min, fnet)  

local fNetRef = fnet:clone()
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
stream1:add(nn.contrastMax2ndMax(dist_min))

-- stream2 :
stream2:add(nn.Narrow(2,1, img_w - 2*hpatch - disp_max))
stream2:add(nn.Transpose({1,2}))
stream2:add(nn.contrastMax2ndMax(dist_min))

Net:add(nn.JoinTable(1))
Net:add(nn.SplitTable(2))
Net:add(nn.JoinTable(1))
Net:add(nn.SplitTable(2))

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
