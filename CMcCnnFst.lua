--[[

This is class implements MC-NN (fst) from 
"Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches" by Jure Zbontar.


Consists of 4 featureNets that share weights and biases:
fNet_1      - corresponds to patch
fNetMatch   - corresponds to "match" patch
fNet_2      - corresponds to patch
fNetNoMatch - corresponds to "no match" patch

Outputs of fNet_1-fNetMatch are feeded to hNetMatch
Outputs of fNet_2-fNetNoMatch are feeded to hNetNoMatch  

hNetMatch and hNetNoMatch compare the inputs using Cosine distance measure:

                        input1.*input2
cos(input1, input2) = --------------------
                      ||input1||*||input2||

The output of hNetMatch and hNetNoMatch are feeded into loss function

]]--

do
  
local mcCnnFst = torch.class('mcCnnFst')

function mcCnnFst:__init( nbConvLayers, nbFeatureMap, kernel, param )
      -- nbConvLayers - number of convolutional layers
      -- nbFeatureMap - number of feature maps in every layer
      -- kernel       - kernel size
      
      self.hook_refPos = torch.Tensor()
      self.hook_refNeg = torch.Tensor() 
      self.hook_posNeg = torch.Tensor()
      
      self.nbConvLayers = nbConvLayers; 
      self.nbFeatureMap = nbFeatureMap;
      self.kernel = kernel;
      
      -- size of equivalent input patch 
      self.patchSize = 1 + nbConvLayers*(kernel - 1); 
      self.hpatch = (self.patchSize - 1) / 2
end



function mcCnnFst:getMilNetDoubleBatch(img_w, disp_max)
  
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

self.hook_refPos = seqRefPos.output:contiguous()
self.hook_refNeg = seqRefNeg.output:contiguous()
self.hook_posNeg = seqPosNeg.output:contiguous()

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
local mask = torch.ones(img_w-2*self.hpatch, img_w-2*self.hpatch)*2  
mask = torch.triu(torch.tril(mask,-1),-disp_max)
mask = mask - 2;
seqRefNegMask:add(nn.addMatrix(mask))
seqRefPosMask:add(nn.addMatrix(mask))
seqRefPosMask:add(nn.Narrow(1,disp_max+1, img_w - 2*self.hpatch - disp_max))
seqRefNegMask:add(nn.Narrow(1,disp_max+1, img_w - 2*self.hpatch - disp_max))
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
local mask = torch.ones(img_w-2*self.hpatch, img_w-2*self.hpatch)*2  
mask = torch.triu(torch.tril(mask,-1),-disp_max)
mask = mask - 2;
seqPosNegMask:add(nn.addMatrix(mask))
seqRefPosMask:add(nn.addMatrix(mask))
seqRefPosMask:add(nn.Narrow(2,1, img_w - 2*self.hpatch - disp_max))
seqPosNegMask:add(nn.Narrow(2,1, img_w - 2*self.hpatch - disp_max))
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


function mcCnnFst:getMilNetBatch(img_w, disp_max)
  
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

function mcCnnFst:getMilNet(img_w)
    
      -- input table with 3 tensors : 
    -- 1 x p_h x p_w - reference patch
    -- 1 x p_h x (epi_w+2*p_w) - match epipolar line
    -- 1 x p_h x (epi_w+2*p_w) - no-match epipolar line
    -- where p_h, p_w are hight and width of reference patch 
    -- and epi_w length of epipolar line                     
    
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
    local module = nn.Squeeze()
    epiNet1:add(module)
    local module = nn.Squeeze()
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
    
    -- PATCH NET
    -- reference patch we pass through feature net and get 64 x 1 x 1 response
    parTab:add(patchNet);
            
    -- squeeze 
    local module = nn.Squeeze()
    patchNet:add(module)
    local module = nn.Unsqueeze(1)
    patchNet:add(module)

    -- normalize
    local module = nn.Normalize(2)
    patchNet:add(module)
    
    -- multiply
    local module = nn.MM(false, true)
    Net:add(module)
    
    -- divide into two col
    local module = nn.View(img_w - 2*self.hpatch, 2)
    Net:add(module)
    
    -- compute max in every col
    local module = nn.Max(1)
    Net:add(module)
    
    -- split
    local module = nn.Unsqueeze(1)
    Net:add(module)
    local module = nn.SplitTable(2)
    Net:add(module)
    
    return Net;
end


function mcCnnFst:getBagNet(nb_pairs)
    
    -- input table with 2 tensors : 
    -- 1 x p_h x p_w - reference patch
    -- 1 x p_h x (epi_w+2*p_w) - epipolar line
    -- where p_h, p_w are hight and width of reference patch 
    -- and epi_w length of epipolar line                                 
        
    local Net = nn.Sequential();
    local parTab = nn.ParallelTable();
    Net:add(parTab)
        
    -- epipolar line we pass through same feature net and get 64 x 1 x epi_w
    local epiNet = model:getFeatureNet();
    parTab:add(epiNet);
    
    -- reference patch we pass through feature net and get 64 x 1 x 1 response
    local patchNet = epiNet:clone('weight','bias', 'gradWeight','gradBias');
    parTab:add(patchNet);
            
    -- squeeze 
    local module = nn.Squeeze()
    patchNet:add(module)
    module = nn.Unsqueeze(1)
    patchNet:add(module)
    local module = nn.Squeeze()
    epiNet:add(module)
    
    -- transpose 
    local module = nn.Transpose({1,2})
    epiNet:add(module)
    
    -- normalize
    local module = nn.Normalize(2)
    epiNet:add(module)
    local module = nn.Normalize(2)
    patchNet:add(module)
    
    -- multiply
    local module = nn.MM(false, true)
    Net:add(module)
    
    -- max
    local module = nn.Max(1)
    Net:add(module)
    
    return Net;
end

-- This function creates basic feature net
-- input of the nextwork is 1 x height x width  tensor
function mcCnnFst:getFeatureNet()
    
  local fNet = nn.Sequential();
  
  for nConvLayer = 1, self.nbConvLayers do

    -- if first layer, then input is just gray image
    -- otherwise input is featuremaps of previous layer
    if( nConvLayer == 1 ) then
      nInputPlane  = 1 
    else
      nInputPlane  = self.nbFeatureMap
    end

    local nOutputPlane = self.nbFeatureMap; -- number of feature maps in layer
    local kW = self.kernel;            -- kernel width and height  
    local kH = self.kernel;
    local dW = 1;                 -- step of convolution 
    local dH = 1;

    local module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH);
    fNet:add(module);

    -- Make ReLU (rectified linear unit)
    -- last convolutional layer does not have ReLU unit
    if( nConvLayer < self.nbConvLayers ) then
      fNet:add(nn.ReLU());
    end

  end 

  -- Reshape output of feature network 
  --local module = nn.Reshape(self.nbFeatureMap);
 -- local module = nn.Squeeze();
 -- fNet:add(module);  

  return fNet

end

-- Function creates Siamese network
-- Input of network is table of 2 of 1 x patch_height x patch_width tensors
-- Output is cosine distance 
function mcCnnFst:getSiameseNet()
    
    local siamNet = nn.Sequential();
    local parNet = nn.ParallelTable();
    
    -- create basis of the triplet net - the feature net
    local fNet1 = self:getFeatureNet();
        
    -- clone feature net, while sharing parameters and gradients
    local fNet2 = fNet1:clone('weight','bias', 'gradWeight','gradBias');
    
    -- add 2 feature nets in parallel table container 
    parNet:add(fNet1)
    parNet:add(fNet2)
    
    -- add parallel table container in sequential container
    siamNet:add(parNet);
    
    -- compute cosine distance
    local module = nn.CosineDistance();
    siamNet:add(module)
    
    return siamNet
    
end  
  
-- Function creates triplet network by combining 3 feature nets
-- Input of network is table of 3 of 1 x patch_height x patch_width 
-- (order is important Patch, Positive, Negative)
-- Output of network is table of 2 numbers
function mcCnnFst:getTripletNet()
      
    local Net = nn.Sequential();
  local parTab = nn.ParallelTable();
  Net:add(parTab);

  -- EPI NETs      
  -- epipolar line we pass through same feature net and get 64 x 1 x epi_w
  local epiSeq = nn.Sequential();
  parTab:add(epiSeq)
  local epiParTab = nn.ParallelTable();
  epiSeq:add(epiParTab);
  local epiNet1 = self:getFeatureNet();
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
  
  return Net
  
end
 
end 
