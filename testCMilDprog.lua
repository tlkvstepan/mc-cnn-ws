require 'gnuplot'
require 'nn'
require 'image'

-- Custom modules
require 'libdprog'                        -- C++ module for dynamic programming
dofile('CMilDprog.lua');                     -- Dynamic programming module


input = torch.r
module = nn.milDprog()

disp_max = 200

mask = torch.triu(torch.tril(torch.ones(1000,1000),-1),-disp_max)

refPos = torch.rand(1000,1000) + torch.eye(1000,1000)
refNeg = torch.rand(1000,1000) + torch.eye(1000,1000)
negPos = torch.rand(1000,1000) + torch.eye(1000,1000)
input = {refPos, refNeg, negPos}
output = module:forward(input)

gradOutput = {}
gradOutput[1] = torch.rand(output[1]:size(1), output[1]:size(2))
gradOutput[2] = torch.rand(output[2]:size(1), output[2]:size(2))

gradInput = module:backward(input, gradOutput)

print(out)
print(inputGrad)