local netWrapper = {}

function netWrapper.getMilContrastDprog(img_w, disp_max, hpatch, dist_min, loss_margin, fnet)

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

-- find best dprog solution for ref-neg and neg-pos
dNetRefNeg:add(nn.Dprog(dist_min))
dNetNegPos:add(nn.Dprog(dist_min))

-- find dprog solution for ref-pos and pos-ref
-- and alternative max solution that is on minimum distance from dprog solution
dNetRefPos:add(nn.contrastDprog(dist_min))
dNetRefPos:add(nn.SplitTable(2))
dNetPosRef:add(nn.contrastDprog(dist_min))
dNetPosRef:add(nn.SplitTable(2))

---- flatten tables hierarchy
---- after flattening, order is following 
---- (1) ref-pos-dprog, (2) ref-pos-max, (3) pos-ref-dprog, (4) pos-ref-max, (5) ref-neg-dprog, (6) pos-neg-dprog
Net:add(nn.FlattenTable())

-- make 4 output tables of tables
local dNet2CostCom = nn.ConcatTable()
Net:add(dNet2CostCom); -- feature net to distance net commutator
local milFwd = nn.Sequential()
local milBwd = nn.Sequential()
local contrastFwd = nn.Sequential()
local contrastBwd = nn.Sequential()
dNet2CostCom:add(milFwd)
dNet2CostCom:add(milBwd)
dNet2CostCom:add(contrastFwd)
dNet2CostCom:add(contrastBwd)
local milFwdSel = nn.ConcatTable()  -- input selectors for cost
local milBwdSel = nn.ConcatTable()
local contrastFwdSel = nn.ConcatTable()  -- input selectors for each distance net
local contrastBwdSel = nn.ConcatTable()
milFwd:add(milFwdSel)
milBwd:add(milBwdSel)
contrastFwd:add(contrastFwdSel)
contrastBwd:add(contrastBwdSel)
milFwdSel:add(nn.SelectTable(1))  -- ref-pos-dprog
milFwdSel:add(nn.SelectTable(5))  -- ref-neg-dprog
milBwdSel:add(nn.SelectTable(3))  -- pos-ref-dprog
milBwdSel:add(nn.SelectTable(6))  -- ref-neg-dprog
contrastFwdSel:add(nn.SelectTable(1))  -- ref-pos-dprog
contrastFwdSel:add(nn.SelectTable(2))  -- ref-pos-max
contrastBwdSel:add(nn.SelectTable(3))  -- ref-pos-dprog
contrastBwdSel:add(nn.SelectTable(4))  -- ref-pos-max

-- define criterion
-- loss(x(+), x(-)) = max(0,  - x(+) + x(-)  + prm['loss_margin'])
local milFwdCst = nn.MarginRankingCriterion(loss_margin);
local milBwdCst = nn.MarginRankingCriterion(loss_margin);
local contrastFwdCst = nn.MarginRankingCriterion(loss_margin);
local contrastBwdCst = nn.MarginRankingCriterion(loss_margin);
local criterion = nn.ParallelCriterion():add(milFwdCst,0.25):add(milBwdCst,0.25):add(contrastFwdCst,0.25):add(contrastBwdCst,0.25)

return Net, criterion

end

function netWrapper.getMilContrastMax(img_w, disp_max, hpatch, dist_min, loss_margin, fnet)

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

dNetRefNeg:add(nn.Max(2))
dNetNegPos:add(nn.Max(2))
dNetRefPos:add(nn.contrastMax(dist_min))
dNetRefPos:add(nn.SplitTable(2))
dNetPosRef:add(nn.contrastMax(dist_min))
dNetPosRef:add(nn.SplitTable(2))

-- flatten tables hierarchy
-- after flattening, order is following 
-- (1) ref-pos-max, (2) ref-pos-maxm, (3) pos-ref-max, (4) pos-ref-maxm, (5) ref-neg-max, (6) pos-neg-max
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
milBwdSel:add(nn.SelectTable(6))  -- pos-neg-max
maxFwdSel:add(nn.SelectTable(1))  -- ref-pos-max
maxFwdSel:add(nn.SelectTable(2))  -- ref-pos-maxm
maxBwdSel:add(nn.SelectTable(3))  -- ref-pos-max
maxBwdSel:add(nn.SelectTable(4))  -- ref-pos-maxm

-- define criterion
-- loss(x(+), x(-)) = max(0,  - x(+) + x(-)  + prm['loss_margin'])
local milFwdCst = nn.MarginRankingCriterion(loss_margin);
local milBwdCst = nn.MarginRankingCriterion(loss_margin);
local maxFwdCst = nn.MarginRankingCriterion(loss_margin);
local maxBwdCst = nn.MarginRankingCriterion(loss_margin);
local criterion = nn.ParallelCriterion():add(milFwdCst,0.25):add(milBwdCst,0.25):add(maxFwdCst,0.25):add(maxBwdCst,0.25)

return Net, criterion

end

function netWrapper.getMilMax(img_w, disp_max, hpatch, loss_margin, fnet)

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

-- compute row-wise maximums
dNetRefPos:add(nn.Max(2))
dNetRefNeg:add(nn.Max(2))
dNetPosRef:add(nn.Max(2))
dNetNegPos:add(nn.Max(2))

-- flatten tables hierarchy
-- after flattening, order is following 
-- ref-pos-max, pos-ref-max, ref-neg-max, pos-neg-max
Net:add(nn.FlattenTable())

-- make 2 output tables of tables
local dNet2CostCom = nn.ConcatTable()
Net:add(dNet2CostCom); -- feature net to distance net commutator
local milFwd = nn.Sequential()
local milBwd = nn.Sequential()
dNet2CostCom:add(milFwd)
dNet2CostCom:add(milBwd)
local milFwdSel = nn.ConcatTable()  -- input selectors for cost
local milBwdSel = nn.ConcatTable()
milFwd:add(milFwdSel)
milBwd:add(milBwdSel)
milFwdSel:add(nn.SelectTable(1))  -- ref-pos-max
milFwdSel:add(nn.SelectTable(3))  -- ref-neg-max
milBwdSel:add(nn.SelectTable(2))  -- pos-ref-max
milBwdSel:add(nn.SelectTable(4))  -- pos-neg-max

local milFwdCst = nn.MarginRankingCriterion(loss_margin);
local milBwdCst = nn.MarginRankingCriterion(loss_margin);
local criterion = nn.ParallelCriterion():add(milFwdCst,1):add(milBwdCst,1)

return Net, criterion

end

function netWrapper.getDistNet(img_w, disp_max, hpatch, fnet)

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

-- mask impossible disparity values in distance mask)
-- (so that all impossible values are equal to -1)
local mask = torch.ones(img_w-2*hpatch, img_w-2*hpatch)*2  
mask = torch.triu(torch.tril(mask,-1),-disp_max)
mask = mask - 2; 
Net:add(nn.addMatrix(mask))
Net:add(nn.Clamp(-1, 1))
   
return Net

end

function netWrapper.getMilDprog(img_w, disp_max, hpatch, loss_margin, fnet)

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

-- compute row-wise maximums
dNetRefPos:add(nn.Dprog())
dNetRefNeg:add(nn.Dprog())
dNetPosRef:add(nn.Dprog())
dNetNegPos:add(nn.Dprog())

-- flatten tables hierarchy
-- after flattening, order is following 
-- ref-pos-max, pos-ref-max, ref-neg-max, pos-neg-max
Net:add(nn.FlattenTable())

-- make 2 output tables of tables
local dNet2CostCom = nn.ConcatTable()
Net:add(dNet2CostCom); -- feature net to distance net commutator
local milFwd = nn.Sequential()
local milBwd = nn.Sequential()
dNet2CostCom:add(milFwd)
dNet2CostCom:add(milBwd)
local milFwdSel = nn.ConcatTable()  -- input selectors for cost
local milBwdSel = nn.ConcatTable()
milFwd:add(milFwdSel)
milBwd:add(milBwdSel)
milFwdSel:add(nn.SelectTable(1))  -- ref-pos-max
milFwdSel:add(nn.SelectTable(3))  -- ref-neg-max
milBwdSel:add(nn.SelectTable(2))  -- pos-ref-max
milBwdSel:add(nn.SelectTable(4))  -- pos-neg-max

local milFwdCst = nn.MarginRankingCriterion(loss_margin);
local milBwdCst = nn.MarginRankingCriterion(loss_margin);
local criterion = nn.ParallelCriterion():add(milFwdCst,1):add(milBwdCst,1)

return Net, criterion

end

function netWrapper.getContrastMax(img_w, disp_max, hpatch, dist_min, loss_margin, fnet)  

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
local splitter = nn.ConcatTable()
Net:add(splitter)
local stream1 = nn.Sequential()
local stream2 = nn.Sequential()
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


local contrastFwdCst = nn.MarginRankingCriterion(loss_margin);
local contrastBwdCst = nn.MarginRankingCriterion(loss_margin);
local criterion = nn.ParallelCriterion():add(contrastFwdCst,1):add(contrastBwdCst,1)

return Net, criterion
end 

function netWrapper.getContrastDprog(img_w, disp_max, hpatch, dist_min, loss_margin, fnet)  

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

-- clamp (-1, 1)
Net:add(nn.Clamp(-1,1))

-- convert range to (0 1)
Net:add(nn.AddConstant(1))
Net:add(nn.MulConstant(0.5))


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
NetBCostSel:add(nn.SelectTable(1))
NetBCostSel:add(nn.SelectTable(3))

local contrastiveFwdCst = nn.MarginRankingCriterion(loss_margin);
local contrastiveBwdCst = nn.MarginRankingCriterion(loss_margin);
local criterion = nn.ParallelCriterion():add(contrastiveFwdCst,1):add(contrastiveBwdCst,1)

--Net:add(nn.JoinTable(1))
--Net:add(nn.SplitTable(2))

return Net, criterion
end 

-- Function creates triplet network by combining 3 feature nets
-- Input of network is table of 3 of 1 x patch_height x patch_width 
-- (order is important Patch, Positive, Negative)
-- Output of network is table of 2 numbers
function netWrapper.getTripletNet(fnet)
      
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
 
return netWrapper
