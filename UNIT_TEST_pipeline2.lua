-- standard modules
require 'gnuplot'
require 'optim'
require 'nn'
require 'image'
require 'torch'
require 'lfs'

-- cuda 
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true
   
-- custom
--dofile('copyElements.lua')
--dofile('fixedIndex.lua')

dofile('CPipeline.lua');         
dofile('CHeadNetMulti.lua')
cnnMetric = dofile('CCNNMetric.lua');       
trainerNet = dofile('CTrainerNet.lua')    
testFun = dofile('CTestUtils.lua');         

disp_max = 10;
width = 100;
arch = 'fst-kitti'
th_sup = 1
th_sup = 1
loss_margin = 1

local embed_net = cnnMetric.getEmbeddNet(arch)
local head_net  = cnnMetric.getHeadNet(arch)
head_net:cuda()
embed_net:cuda()
--siam_net  = nnMetric.setupSiamese(embed_net, head_net, width, disp_max)

tr_net, criterion = trainerNet.getPipeline(disp_max, width, th_sup, loss_margin, embed_net, head_net)
tr_net:cuda()
criterion:cuda()
      
hpatch = cnnMetric.getHPatch(embed_net);

ref = torch.rand(1, hpatch*2+1, width ):cuda() 
pos = torch.rand(1, hpatch*2+1, width ):cuda()

matchInRow = torch.round(torch.rand(width)*(width-2*hpatch)):cuda() + hpatch
matchInRow[53] = 1/0
matchInRow[23] = 1/0
matchInRow[75] = 1/0
matchInRow[76] = 1/0
matchInRow = matchInRow[{{hpatch+1, width-hpatch}}]
input  = {{ref, pos}, matchInRow};

tr_net:forward(input)

target = {torch.ones(tr_net.output[1][1]:size()):cuda(), torch.ones(tr_net.output[1][1]:size()):cuda()}

criterion:forward(tr_net.output, target)

tr_net:backward(input, criterion:backward(tr_net.output, target))



