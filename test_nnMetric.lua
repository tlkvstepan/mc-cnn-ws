require 'torch'
require 'nn'
require 'cunn'
require 'gnuplot'

dofile('copyElements.lua')
dofile('fixedIndex.lua')
include('../mc-cnn/SpatialConvolution1_fw.lua')

nnMetric = dofile('nnMetric.lua')

embeddNet = nnMetric.getEmbeddNet('fst-kitti')
headNet = nnMetric.getHeadNet('fst-kitti')


--headNet:replace(function(module)
--   if torch.typename(module) == 'nn.Linear' then
        
--        local weight = module.weight;
--        local bias = module.bias;
--        local nInputPlane  = module.weight:size(2)  
--        local nOutputPlane = module.weight:size(1)  
--        local substitute = nn.SpatialConvolution1_fw(nInputPlane, nOutputPlane)
--        substitute.weight:copy(weight)
--        substitute.bias:copy(bias)
--        return substitute
--   else
--      return module
--   end
--  end)


----embeddNet_pad = nnMetric.padBoundary(embeddNet)


hpatch = nnMetric.getHPatch(embeddNet)


--parametric = nnMetric.isParametric(headNet)

width  = 1000
dispMax = 300

siamNet = nnMetric.setupSiamese(embeddNet, headNet, width, dispMax)

siamNet:cuda()
embedNet_, headNet_ = nnMetric.parseSiamese(siamNet)
print(siamNet)

input = {torch.rand(1, hpatch*2+1, width):cuda(), torch.rand(1, hpatch*2+1, width):cuda()}
outputGrad = torch.rand(width-hpatch*2, width-hpatch*2):cuda();

siamNet:forward(input)
a = 1
--inputGrad = siamNet:backward(input, outputGrad)

--gnuplot.imagesc(siamNet.output,'gray')







