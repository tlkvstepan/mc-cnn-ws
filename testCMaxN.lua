require 'nn'
dofile('CMaxM.lua')

gradOutput = torch.rand(2)
input = torch.Tensor{{1,2,3},{4,3,12}}
moduleMaxN = nn.MaxM(2,2)
moduleMax = nn.Max(2)

outputMaxM = moduleMaxN:forward(input)
gradInputMaxM = moduleMaxN:updateGradInput(input, gradOutput)

outputMax = moduleMax:forward(input)
gradInputMax = moduleMax:updateGradInput(input, gradOutput)

print(output)