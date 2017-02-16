--standard modules
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
dofile('CMyNN.lua')
--dofile('copyElements.lua')
--dofile('fixedIndex.lua')

dofile('CMil.lua')
dofile('CContrastive.lua')
dofile('CHeadNetMulti.lua')
cnnMetric = dofile('CCNNMetric.lua');       
trainerNet = dofile('CTrainerNet.lua')    
testFun = dofile('CTestUtils.lua');         

disp_max = 10;
width = 100;
arch = 'fst-kitti'
loss_margin = 0.2

--embed_net = cnnMetric.getEmbeddNet(arch)
--head_net  = cnnMetric.getHeadNet(arch)
head_net = cnnMetric.cosHead(10);
head_net:cuda()

ref = torch.randn(100, 10):cuda() 
neg = torch.randn(100, 10):cuda() 
input = {ref, neg};
head_net:forward(input)

x = 10
