local nnMetric = {}
 
--------------------------------------------------------------------------------- 
------------------- User interface functions ------------------------------------
---------------------------------------------------------------------------------

-- Function returns embedding net given network name 
function nnMetric.getEmbeddNet(metricName)
    
  assert( metricName == 'fst-mb' or
          metricName == 'fst-kitti' or
          metricName == 'acrt-mb' or
          metricName == 'acrt-kitti', 'wrong network name!') 
  
  local nbConvLayers   
  local nbFeatureMap 
  local kernel 
  
  if( metricName == 'fst-mb' ) then
  
    nbConvLayers = 5  
    nbFeatureMap = 64
    kernel = 3
  
  elseif( metricName == 'acrt-mb' ) then
  
    nbConvLayers = 5  
    nbFeatureMap = 112
    kernel = 3
  
  elseif( metricName == 'fst-kitti' ) then
    
    nbConvLayers = 4  
    nbFeatureMap = 64
    kernel = 3
        
  elseif( metricName == 'acrt-kitti' ) then
    
    nbConvLayers = 4  
    nbFeatureMap = 112
    kernel = 3
      
  end
      
  local embedNet = nnMetric.embeddNet( nbConvLayers, nbFeatureMap, kernel )
  
  --if ( metricName == 'acrt-mb' or metricName == 'acrt-kitti' ) then
  --  embedNet:add( nn.ReLU() )         -- add nonlinearity to last layer of embed net for accurate architecture
  --end
  
  return embedNet

end

-- Function returns head net given network name 
function nnMetric.getHeadNet(metricName)
    
  assert( metricName == 'fst-mb' or
          metricName == 'fst-kitti' or
          metricName == 'acrt-mb' or
          metricName == 'acrt-kitti', 'wrong network name!') 
  
  local headNet 
  
  if( metricName == 'fst-mb' ) then
  
    local nbFeatureMap = 64
    
    headNet = nnMetric.cosHead(nbFeatureMap)
    
  elseif( metricName == 'acrt-mb' ) then
    
    local nbFeatureMap  = 112
    local nbFcLayers    = 3
    local nbFcUnits     = 384
    
    headNet = nnMetric.fcHead(nbFeatureMap, nbFcLayers, nbFcUnits)
  
  elseif( metricName == 'fst-kitti' ) then
  
    local nbFeatureMap = 64
    
    headNet = nnMetric.cosHead(nbFeatureMap)
        
  elseif( metricName == 'acrt-kitti' ) then
    
    local nbFeatureMap  = 112
    local nbFcLayers    = 4
    local nbFcUnits     = 384
        
    headNet = nnMetric.fcHead(nbFeatureMap, nbFcLayers, nbFcUnits)
    
  end
      
  return headNet

end


-- Function parse siamese net into embedding net and head net
function nnMetric.parseSiamese(siamNet) 
    
  local embedNet = siamNet.modules[1].modules[1]:clone()  
  embedNet:remove(8) -- delete squeeze and transpose
  embedNet:remove(8)
  
    
  local headNet =  siamNet.modules[3]:clone()  
  
  return embedNet, headNet
  
end

-- Function sets up siamese net 
function nnMetric.setupSiamese(embedNet, headNet, width, dispMax)

local siamNet =  nn.Sequential()

local hpatch = nnMetric.getHPatch(embedNet)

local  activeRows 
local  activeCols
local  activeIdx
local  nbActivePairs

---------------------- Find active elementes of similarity matrix --------------------
do 
    
  local row = torch.Tensor(width-hpatch*2, 1)
  row[{{},{1}}] = torch.range(1, width-hpatch*2)
  local rows = torch.repeatTensor(row, 1, width-hpatch*2)

  local col = row:t():clone()
  local cols = torch.repeatTensor(col, width-hpatch*2, 1)

  local disp = rows - cols 
  local mask = disp:le(dispMax):cmul( disp:gt(0) )

  activeIdx = (cols-1) + (rows-1)*(width-hpatch*2) + 1
  activeIdx = activeIdx[mask] 

  activeCols = cols[mask]
  activeRows = rows[mask]

  nbActivePairs = mask:ne(0):sum()

end

------------------------------------------------------------------

local twoEmbedNet 

---------------------- Make two embeded nets -------------------------------------------
do
    
  twoEmbedNet =  nn.ParallelTable()
  
  -- make two towers
  local embedNet0 = embedNet:clone()

  -- nb_features x 1 x width-hpatch*2 ==> width-hpatch*2 x nb_features
  embedNet0:add(nn.Squeeze(2))
  embedNet0:add(nn.Transpose({1,2}))

  -- second tower is clone of the first one
  local embedNet1 = embedNet0:clone('weight','bias', 'gradWeight','gradBias');

  twoEmbedNet:add(embedNet0)
  twoEmbedNet:add(embedNet1)

end



----------------------------------------------------------------------------------------

siamNet:add(twoEmbedNet)

local pairSelNet

----------------------- Make two pair selecting nets ------------------------------------
do 
  
  pairSelNet =  nn.ParallelTable()
  
  pairSelNet:add(nn.fixedIndex(1, activeRows:long()))
  pairSelNet:add(nn.fixedIndex(1, activeCols:long()))

end

-----------------------------------------------------------------------------------------

siamNet:add(pairSelNet)

siamNet:add(headNet)

--siamNet:add( nn.Squeeze(2) )


siamNet:add(nn.copyElements(torch.LongStorage{nbActivePairs}, torch.LongStorage{width-hpatch*2, width-hpatch*2}, torch.range(1, nbActivePairs), activeIdx))

  return siamNet 
end


--function nnMetric.getTestNet(net)
  
--   for i =  1,#net.modules do
--    if torch.typename(net.modules[i]) == 'nn.SpatialConvolution' or
--       torch.typename(net.modules[i]) == 'cudnn.SpatialConvolution' then
       
--       local nInputPlane  = net.modules[i].nInputPlane  
--       local nOutputPlane = net.modules[i].nInputPlane  
       
--    end
--  end
  
  
--  net:replace(function(module)
--   if torch.typename(module) == 'nn.SpatialConvolution' then
--        local nInputPlane  = module.nInputPlane  
--        local nOutputPlane = module.nOutputPlane  
--        local weight = module.weight;
--        local bias = module.weight;
--        local substitute = nn.SpatialConvolution1_fw(nInputPlane, nOutputPlane))
--        substitute.weight:copy(weight)
--        substitute.bias:copy(bias)
--        return substitute
--   else
--      return module
--   end
--  end)

----  return net;
  
  
----  net_te_all =  {}
----  for i, v in ipairs(net_te.modules) do table.insert(net_te_all, v) end
----  for i, v in ipairs(net_te2.modules) do table.insert(net_te_all, v) end

----  local finput = torch.CudaTensor()
----  local i_tr = 1
----  local i_te = 1
----  while i_tr <= net_tr:size() do
----    local module_tr = net_tr:get(i_tr)
----    local module_te = net_te_all[i_te]

----    local skip = {['nn.Reshape']=1, ['nn.Dropout']=1}
----    while skip[torch.typename(module_tr)] do
----      i_tr = i_tr + 1
----      module_tr = net_tr:get(i_tr)
----    end

----    if module_tr.weight then
----      -- print(('tie weights of %s and %s'):format(torch.typename(module_te), torch.typename(module_tr)))
----      assert(module_te.weight:nElement() == module_tr.weight:nElement())
----      assert(module_te.bias:nElement() == module_tr.bias:nElement())
----      module_te.weight = torch.CudaTensor(module_tr.weight:storage(), 1, module_te.weight:size())
----      module_te.bias = torch.CudaTensor(module_tr.bias:storage(), 1, module_te.bias:size())
----    end

----    i_tr = i_tr + 1
----    i_te = i_te + 1
----  end
      
  
  
--end


function nnMetric.padBoundary(net)
  for i =  1,#net.modules do
    if torch.typename(net.modules[i]) == 'nn.SpatialConvolution' then
      net.modules[i].padW = 1
      net.modules[i].padH = 1
    end
  end
  return net;
end

function nnMetric.isParametric(net)
   for i = 1,#net.modules do
      local module = net:get(i)
      if torch.typename(module) == 'cudnn.SpatialConvolution' or
         torch.typename(module) == 'nn.SpatialConvolution' or 
         torch.typename(module) == 'nn.Linear' or 
         torch.typename(module) == 'cudnn.Linear' then
         return true;
      end
   end
   return false
end


function nnMetric.getHPatch(embeddNet)
   local ws = 1
   for i = 1,#embeddNet.modules do
      local module = embeddNet:get(i)
      if torch.typename(module) == 'cudnn.SpatialConvolution' or torch.typename(module) == 'nn.SpatialConvolution' then
         ws = ws + module.kW - 1
      end
   end
   return (ws-1)/2
end

----------------------------------------------------------------
------------------------------ embeding net --------------------
----------------------------------------------------------------

-- Given tensor 1 x hpatch*2 x width embedding net produces feature 
-- tensor of size 64 x 1 x width-hpatch*2 

function nnMetric.embeddNet( nbConvLayers, nbFeatureMap, kernel )     
    
  local fNet = nn.Sequential();
  
  for nConvLayer = 1, nbConvLayers do

    -- if first layer, then input is just gray image
    -- otherwise input is featuremaps of previous layer
    local nInputPlane
    if( nConvLayer == 1 ) then
      nInputPlane  = 1 
    else
      nInputPlane  = nbFeatureMap
    end

    local nOutputPlane = nbFeatureMap; -- number of feature maps in layer
    local kW = kernel;            -- kernel width and height  
    local kH = kernel;
    local dW = 1;                 -- step of convolution 
    local dH = 1;
    
    padW = 0;
    padH = 0;
      
    local module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH);
    fNet:add(module);

    -- Make ReLU (rectified linear unit) last convolutional layer does not have ReLU unit
    if( nConvLayer < nbConvLayers ) then
      fNet:add(nn.ReLU());
    end

  end 
    
  local patchSize = 1 + nbConvLayers*(kernel - 1); 
  local hpatch = (patchSize - 1) / 2
      
  return fNet 

end

----------------------------------------------------------------
------------------------------ heads ---------------------------
----------------------------------------------------------------

-- Given two tensors of size nb_pairs x nb_features, head network computes
-- distance tensor of size nb_pairs 

-- fully connected linear net
function nnMetric.fcHead(nbFeatureMap, nbFcLayers, nbFcUnits)

  local fcHead =  nn.Sequential()
  fcHead:add( nn.JoinTable(2) )
  fcHead:add( nn.ReLU() )         -- add nonlinearity to last layer of embed net for accurate architecture
  for nFcLayer = 1,nbFcLayers do
     local idim = (nFcLayer == 1) and nbFeatureMap*2 or nbFcUnits
     local odim = nbFcUnits
     fcHead:add( nn.Linear(idim, odim) )
     fcHead:add( nn.ReLU(true) )
  end
  fcHead:add( nn.Linear(nbFcUnits, 1) )
  fcHead:add( nn.Sigmoid(true) )
  
  return fcHead
end

-- cosine head
function nnMetric.cosHead(nbFeatureMap)
  
  local cosNet = nn.Sequential()
  local normNet = nn.ParallelTable()
  cosNet:add(normNet)
  normNet:add(nn.Normalize(2))
  normNet:add(nn.Normalize(2))
  cosNet:add(nn.DotProduct())
    
  -- convert range to (0 1)
  cosNet:add(nn.AddConstant(1))
  cosNet:add(nn.MulConstant(0.5))
      
  return cosNet
  
end

return nnMetric