require 'torch'
require 'nn'
dofile('CContrastMax2ndMax.lua')


input = torch.rand(1000,1000)
outputGrad = torch.rand(3,2)
local max2ndMax = nn.contrastMax2ndMax(1)

out = max2ndMax:forward(input)
inputGrad = max2ndMax:backward(input,outputGrad)

print(out)
print(inputGrad)