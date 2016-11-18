require 'nn'
require 'cunn'

nnMetric = dofile('nnMetric.lua')


local nbConvLayers = 5  -- NB!
local nbFeatureMap = 64
local kernel = 3

--local embeddNet = nnMetric.mccnnEmbeddNet( nbConvLayers, nbFeatureMap, kernel )
  
--local headNet = nnMetric.mccnnCosineHead()

--local metricNet = nnMetric.setupSiamese(headNet, embeddNet)
  
metricNet =  nnMetric.get('mc-cnn-fst-kitti')

local embeddNet_, headNet_ = nnMetric.parseSiamese(metricNet:cuda():clone():double())


out = metricNet:forward{torch.rand(1,11,100), torch.rand(1,11,100)}  
  
x = 'stop'  