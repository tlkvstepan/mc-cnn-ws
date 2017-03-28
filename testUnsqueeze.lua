require 'nn'  
require 'cunn'
  net = nn.Sequential()
  net:add(nn.Unsqueeze(1))
  net:cuda()
  x = net:forward(torch.rand(3,4):cuda())
  
  x = 2
  