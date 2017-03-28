require 'torch'
require 'nn'
require 'cunn'
require 'gnuplot'

dofile('copyElements.lua')
dofile('fixedIndex.lua')
include('../mc-cnn/SpatialConvolution1_fw.lua')

width  = 1000
dispMax = 300


nnMetric = dofile('nnMetric.lua')
nnMetric_old = dofile('nnMetric_old.lua')


embeddNet_ = nnMetric.getEmbeddNet('fst-kitti')
headNet_ = nnMetric.getHeadNet('fst-kitti')
siamNet_ = nnMetric.setupSiamese(embeddNet_, headNet_, width, dispMax)
embeddNet, headNet = nnMetric.parseSiamese(siamNet_)
siamNet_ = nnMetric.setupSiamese(embeddNet_, headNet_, width, dispMax)

siamNet_new:cuda()
param_new, grad_new = siamNet_new:getParameters()
grad_new:zero()

siamNet_old = nnMetric_old.get('mc-cnn-fst-kitti')
siamNet_old:cuda()
param_old, grad_old  = siamNet_old:getParameters()
grad_old:zero()

param_new:copy(param_old);
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


--hpatch = nnMetric.getHPatch(embeddNet)


--parametric = nnMetric.isParametric(headNet)

--embedNet_, headNet_ = nnMetric.parseSiamese(siamNet)
--print(siamNet)

input = {torch.rand(1, hpatch*2+1, width):cuda(), torch.rand(1, hpatch*2+1, width):cuda()}

local mask = torch.ones(width-2*hpatch, width-2*hpatch)*1  
mask = torch.triu(torch.tril(mask,-1),-dispMax):cuda()
outputGrad = torch.rand(width-2*hpatch, width-2*hpatch):cuda()
outputGrad = outputGrad:cmul(mask)

siamNet_old:forward(input)
siamNet_new:forward(input)

siamNet_new:backward(input, outputGrad)
siamNet_old:backward(input, outputGrad)

gnuplot.figure(1);
gnuplot.imagesc(torch.abs(siamNet_new.output),'color')


gnuplot.plot(torch.abs(grad_old-grad_new),'color')
a = 1
--gnuplot.imagesc(siamNet.output,'gray')







