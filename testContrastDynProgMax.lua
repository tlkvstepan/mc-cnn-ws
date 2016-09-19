require 'nn'
require 'libdynprog'

dofile('CContrastDynProgMax.lua')

input = torch.rand(1006,1034)
outputGrad = torch.rand(3,2)
local max2ndMax = nn.contrastDynProgMax(1)

out = max2ndMax:forward(input)
inputGrad = max2ndMax:backward(input,outputGrad)

print(out)
print(inputGrad)