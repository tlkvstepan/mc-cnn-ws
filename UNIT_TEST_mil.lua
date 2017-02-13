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

embed_net = cnnMetric.getEmbeddNet(arch)
head_net  = cnnMetric.getHeadNet(arch)

embed_net:cuda()
head_net:cuda()

embed_par, embed_grad = embed_net:getParameters()
head_par, head_grad = head_net:getParameters()

tr_net, criterion = trainerNet.getMil(disp_max, width, loss_margin, embed_net, head_net)  
tr_net:cuda()
criterion:cuda()

hpatch = cnnMetric.getHPatch(embed_net);


-- test head
--desc1 = torch.rand(100,112):cuda()
--desc2 = torch.rand(100,112):cuda()
--head_net:forward{desc1, desc2}

print(embed_grad[{{1,10}}])

ref = torch.randn(1, hpatch*2+1, width ):cuda() 
neg = torch.randn(1, hpatch*2+1, width ):cuda() 
pos = torch.randn(1, hpatch*2+1, width ):cuda()
input  = {ref, neg, pos};
target = {torch.ones(width-disp_max-hpatch*2):cuda(),
          torch.ones(width-disp_max-hpatch*2):cuda(),
          torch.ones(width-disp_max-hpatch*2):cuda()}

tr_net:forward(input)
loss = criterion:forward(tr_net.output, target)
criterion:backward(tr_net.output, target)
tr_net:backward(input, criterion.gradInput )

print(embed_grad[{{1,10}}])



