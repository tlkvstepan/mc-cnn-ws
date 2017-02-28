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
dofile('CPipeline.lua')

-- PARAM
width = 10;
th_sup = 1;


PIPE_NET = nn.pipeline(th_sup)

simMat = torch.rand( width, width ):cuda() 
matchInRow = torch.round(torch.rand(10)*10):cuda()
matchInRow[4] = 1/0
matchInRow[1] = 1/0

PIPE_NET:forward({simMat, matchInRow})

ograd_pipe1 = torch.rand(PIPE_NET.output[1][1]:size()):cuda()
ograd_maxInRow = torch.rand(PIPE_NET.output[1][1]:size()):cuda()
ograd_pipe2 = torch.rand(PIPE_NET.output[1][1]:size()):cuda()
ograd_maxInCol = torch.rand(PIPE_NET.output[1][1]:size()):cuda()

PIPE_NET:backward({simMat, matchInRow}, {{ograd_pipe1, ograd_maxInRow}, {ograd_pipe2, ograd_maxInCol}})


--loss = criterion:forward(tr_net.output, target)
--criterion:backward(tr_net.output, target)
--tr_net:backward(input, criterion.gradInput )

--print(embed_grad[{{1,10}}])



