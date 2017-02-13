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

dofile('CContrastive.lua');         
dofile('CHeadNetMulti.lua')
cnnMetric = dofile('CCNNMetric.lua');       
trainerNet = dofile('CTrainerNet.lua')    
testFun = dofile('CTestUtils.lua');         

disp_max = 10;
width = 100;
arch = 'fst-kitti'
th_occ = 1

embed_net = cnnMetric.getEmbeddNet(arch)
head_net  = cnnMetric.getHeadNet(arch)
head_net:cuda()
embed_net:cuda()
--siam_net  = nnMetric.setupSiamese(embed_net, head_net, width, disp_max)

tr_net, criterion = trainerNet.getContrastive(disp_max, width, th_occ, loss_margin, embed_net, head_net)  
tr_net:cuda()
criterion:cuda()
      
hpatch = cnnMetric.getHPatch(embed_net);

ref = torch.randn(1, hpatch*2+1, width ):cuda() 
neg = torch.randn(1, hpatch*2+1, width ):cuda() 
pos = torch.randn(1, hpatch*2+1, width ):cuda()
input  = {ref, neg, pos};
target = {torch.ones(1, width-disp_max-hpatch*2):cuda(), torch.ones(1, width-disp_max-hpatch*2):cuda()}

tr_net:forward(input)
criterion:forward(tr_net.output, target)
tr_net:backward(input, criterion:backward(tr_net.output, target))


