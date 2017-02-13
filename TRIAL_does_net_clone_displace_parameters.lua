require 'torch'
require 'cunn'
require 'cudnn'

net = nn.Sequential();
net:add(nn.Linear(10, 5))

net:cuda()  -- 10 inputs, 5 outputs
cudnn.convert(net, cudnn)
      
par, grad = net:getParameters()      

grad[1] =0.3
print(grad[{{1,10}}]);
  
net:cuda()  -- 10 inputs, 5 outputs
cudnn.convert(net, cudnn)
  
clone_net2 = net:clone('weight','bias', 'gradWeight','gradBias');
clone_net1 = clone_net2:clone('weight','bias', 'gradWeight','gradBias');


clone_net1:cuda()  -- 10 inputs, 5 outputs
cudnn.convert(clone_net1, cudnn)


print(grad[{{1,10}}]);

input = torch.rand(10):cuda()
clone_net1:forward(input);
clone_net1:backward(input, torch.rand(5):cuda());

print(grad[{{1,10}}]);

x = 2