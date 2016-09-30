require 'nn'
require 'cunn'
require 'libdynprog'
require 'image'

dofile('CDynProg.lua')

dist_min = 2;
local input = image.load('dprog_test.png',1,'byte'):float()
dyProg_module = nn.dynProg(dist_min)

input = torch.rand(1000,1500)
gradOutput = torch.rand(1000)

output = dyProg_module:forward(input)
gradInput = dyProg_module:backward(input, gradOutput)

print(output)