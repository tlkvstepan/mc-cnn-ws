
local baseNet = {}

function baseNet.get( nbConvLayers, nbFeatureMap, kernel, pad_on)
    
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
    
    if pad_on == nil then
      padW = 0;
      padH = 0;
    else
      padW = (kW-1)/2;
      padH = (kH-1)/2;
    end
      
    local module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH);
    fNet:add(module);

    -- Make ReLU (rectified linear unit)
    -- last convolutional layer does not have ReLU unit
    if( nConvLayer < nbConvLayers ) then
      fNet:add(nn.ReLU());
    end

  end 
  
  patchSize = 1 + nbConvLayers*(kernel - 1); 
  hpatch = (patchSize - 1) / 2
      
  return fNet, hpatch 

end

return baseNet