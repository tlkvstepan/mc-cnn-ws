

require 'nn'

net = nn.ConcatTable()
net:add(nn.Identity())
net:add(nn.Identity())

input = torch.rand(3)
out = net:forward(input)

gradOut = {torch.rand(3), torch.rand(3)}
gradIn = net:backward(input, gradOut)
print(out)