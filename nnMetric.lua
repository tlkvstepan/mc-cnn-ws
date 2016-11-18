local nnMetric = {}

function nnMetric.setupSiamese(headNet, embedNet)
  
  local metricNet = nn.Sequential();
  local parTab = nn.ParallelTable();
  metricNet:add(parTab)
   
  local embedNet1 = embedNet:clone();
  local embedNet2 = embedNet1:clone('weight','bias', 'gradWeight','gradBias');

  parTab:add(embedNet1)
  parTab:add(embedNet2)
   
  metricNet:add(headNet:clone())
  
  return metricNet
  
end
 
function nnMetric.parseSiamese(metricNet) 
    
  local embedNet = metricNet.modules[1].modules[1]:clone()  
  local headNet =  metricNet.modules[2]:clone()  
  
  return embedNet, headNet
end

function nnMetric.get(metricName)
    
  assert( metricName == 'mc-cnn-fst-mb' or
          metricName == 'mc-cnn-fst-kitti' or
          metricName == 'mc-cnn-acrt-mb' or
          metricName == 'mc-cnn-acrt-kitti', 'wrong network name!') 
  
  local headNet, embedNet
  if( metricName == 'mc-cnn-fst-mb' ) then 
      
    -- embedding network  
    local nbConvLayers = 5  
    local nbFeatureMap = 64
    local kernel = 3
    embedNet = nnMetric.mccnnEmbeddNet( nbConvLayers, nbFeatureMap, kernel )
  
    -- head network
    headNet = nnMetric.mccnnCosineHead( )
    hpatch = 5
    
  elseif metricName == 'mc-cnn-fst-kitti' then
    
    -- embedding network  
    local nbConvLayers = 4  
    local nbFeatureMap = 64
    local kernel = 3
    embedNet = nnMetric.mccnnEmbeddNet( nbConvLayers, nbFeatureMap, kernel )
  
    -- head network
    headNet = nnMetric.mccnnCosineHead()
    hpatch = 4
        
  else
    
    -- TO-DO
    
  end
  
  -- make metric net
  local metricNet = nnMetric.setupSiamese(headNet, embedNet)
  
  return metricNet, hpatch
end

function nnMetric.mccnnEmbeddNet( nbConvLayers, nbFeatureMap, kernel )     
    
  local fNet = nn.Sequential();
  
  for nConvLayer = 1, nbConvLayers do

    -- if first layer, then input is just gray image
    -- otherwise input is featuremaps of previous layer
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

    -- Make ReLU (rectified linear unit)
    -- last convolutional layer does not have ReLU unit
    if( nConvLayer < nbConvLayers ) then
      fNet:add(nn.ReLU());
    end

  end 
    
  local patchSize = 1 + nbConvLayers*(kernel - 1); 
  local hpatch = (patchSize - 1) / 2
      
  return fNet, hpatch 

end

------------------------------ heads ---------------------------
-- fully connected head
function nnMetric.mccnnFcHead(nbFcUnits, nbLayer)

  -- TO-DO

  return headNet
end

-- cosine head
function nnMetric.mccnnCosineHead()
  
  local headNet = nn.Sequential();
  local normNet1 = nn.Sequential();
  
  local parTab = nn.ParallelTable();
  headNet:add(parTab)
    
  normNet1:add(nn.Squeeze(2))
  normNet1:add(nn.Transpose({1,2}))
  normNet1:add(nn.Normalize(2))
  
  local normNet2 = normNet1:clone()  
  parTab:add(normNet1)
  parTab:add(normNet2)
  
  headNet:add(nn.MM(false, true))

  return headNet
end

return nnMetric