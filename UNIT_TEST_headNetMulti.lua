-- standard modules
require 'gnuplot'
require 'optim'
require 'nn'
require 'image'
require 'torch'

-- cuda 
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true
   
-- custom
dofile('CHeadNetMulti.lua')


cnnMetric = dofile('CCNNMetric.lua');       
trainerNet = dofile('CTrainerNet.lua')    
testFun = dofile('CTestUtils.lua');         

disp_max = 10;
width = 100;
arch = 'acrt-kitti'
loss_margin = 0.2

head_net  = cnnMetric.getHeadNet(arch)

arr_idx = torch.tril(torch.rand(5,5)):nonzero()

headNetMulti = nn.headNetMulti(arr_idx, head_net)
headNetMulti:cuda()
arr1 = torch.randn(5,112):cuda()
arr2 = torch.randn(5,112):cuda()

mask = torch.tril(torch.rand(5,5))
gradOutput = torch.rand(5,5)
gradOutput[mask:le(0.5)] = 0
gradOutput= gradOutput:cuda()

param, grad  = headNetMulti:getParameters()

headNetMulti:forward({arr1, arr2})
headNetMulti:backward({arr1, arr2}, gradOutput)

headNetMulti:forward({arr1, arr2})
headNetMulti:backward({arr1, arr2}, gradOutput)


x = 2


