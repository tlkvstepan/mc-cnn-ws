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
      
      self.nbConvLayers = nbConvLayers; 
      self.nbFeatureMap = nbFeatureMap;
      self.kernel = kernel;
      
      -- size of equivalent input patch 
      self.patchSize = 1 + nbConvLayers*(kernel - 1); 
      self.hpatch = (self.patchSize - 1) / 2
end

function mcCnnFst:getMilNetBatch(img_w, disp_max)
-- input table with 3 tensors : 
-- 1 x p_h x (epi_w - disp_max) - reference epipolar stripe
-- 1 x p_h x (epi_w) - match epipolar line
-- 1 x p_h x (epi_w) - no-match epipolar line
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
local epiNet1 = self:getFeatureNet();
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


local module = nn.Unsqueeze(1)
epiSeq:add(module)

-- PATCH NET
-- reference patch we pass through feature net and get 64 x 1 x 1 response
parTab:add(patchNet);

-- squeeze 
local module = nn.Squeeze()
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

-- divide into two col
local module = nn.Transpose({1,3})
Net:add(module)

---- squeeze
local module = nn.Squeeze()
Net:add(module)

-- make mask
mask = 2*torch.ones(img_w-2*self.hpatch-disp_max,img_w-2*self.hpatch)  
mask = torch.triu(torch.tril(mask, disp_max))
mask = mask - 2;
mask = torch.repeatTensor(mask, 1, 2);
local module = nn.addMatrix(mask)
Net:add(module)

---- unsqueeze
local module = nn.Unsqueeze(3)
Net:add(module)

--- view
local module = nn.View(img_w-2*self.hpatch-disp_max,2,img_w-2*self.hpatch)
Net:add(module)

-- compute max in every col
module = nn.Max(3)
Net:add(module)

-- split
module = nn.SplitTable(2)
Net:add(module)
 
return Net;
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
