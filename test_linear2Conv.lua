require 'torch'
require 'nn'
require 'cudnn'
require 'cutorch'
require 'cunn'
require 'gnuplot'

dofile('copyElements.lua')
dofile('fixedIndex.lua')
include('../mc-cnn/SpatialConvolution1_fw.lua')

width  = 1000
dispMax = 300


nnMetric = dofile('nnMetric.lua')

embedNet = nnMetric.getEmbeddNet('acrt-kitti')



headNet_lin = nnMetric.getHeadNet('acrt-kitti')
headNet_con = headNet_lin:clone();

-- replace linear module with convolution
headNet_con:replace(function(module)
    if torch.typename(module) == 'nn.Linear' then

      local weight = module.weight;
      local bias = module.bias;
      local nInputPlane  = module.weight:size(2)  
      local nOutputPlane = module.weight:size(1)  
      local substitute = nn.SpatialConvolution1_fw(nInputPlane, nOutputPlane)
      substitute.weight:copy(weight)
      substitute.bias:copy(bias)
      return substitute
    else
      return module
    end
  end) 


headNet_con:cuda()
headNet_lin:cuda()


in1 = torch.rand(1,112,1,1):cuda();
in2 = torch.rand(1,112,1,1):cuda();
input_con = {in1, in2 };
input_lin = {nn.utils.addSingletonDimension(in1:squeeze(),1), nn.utils.addSingletonDimension(in2:squeeze(),1)};


headNet_lin:forward(input_lin)
headNet_con:forward(input_con)


--gnuplot.plot(torch.abs(grad_old-grad_new),'color')
a = 1
--gnuplot.imagesc(siamNet.output,'gray')







