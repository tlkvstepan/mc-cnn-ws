require 'torch'
require 'cunn'

net = nn.Sequential();
module1 = nn.Linear(10, 5):cuda()  -- 10 inputs, 5 outputs
module2 = nn.Linear(5, 3)   -- 5 inputs, 3 outputs

net:add(module1);
net:add(nn.Copy('torch.CudaTensor', 'torch.DoubleTensor'))
net:add(module2);


input = torch.rand(10):cuda()
output = net:forward(input);

x = 10

