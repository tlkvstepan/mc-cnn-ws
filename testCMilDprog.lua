require 'gnuplot'
require 'nn'
require 'image'

-- Custom modules
require 'libdprog'                        -- C++ module for dynamic programming
dofile('CMilDprog.lua');                     -- Dynamic programming module


input = torch.r
module = nn.milDprog()

posRef = nn.utils.addSingletonDimension(image.load('dist_mat1.png',1,'byte'),1)
posNeg = nn.utils.addSingletonDimension(image.load('dist_mat2.png',1,'byte'),1)
negPos = nn.utils.addSingletonDimension(image.load('dist_mat3.png',1,'byte'),1)


output = module:forward({posRef, posNeg, negPos})

print(out)
print(inputGrad)