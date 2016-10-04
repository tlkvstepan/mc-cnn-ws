require 'gnuplot'
require 'nn'
require 'image'

-- Custom modules
require 'libdprog'                        -- C++ module for dynamic programming
dofile('CMilDprog.lua');                     -- Dynamic programming module


input = torch.r
module = nn.milDprog()

disp_max = 200


start_cpu = os.time()

for i = 1,300
  refPos = torch.rand(1000,1000)
  refNeg = torch.rand(1000,1000)
  negPos = torch.rand(1000,1000)
  input = {refPos:cuda(), refNeg:cuda(), negPosrefNeg:cuda()}
  output = module:forward(input)

gradOutput = {}
gradOutput[1] = torch.rand(output[1]:size(1), output[1]:size(2))
gradOutput[2] = torch.rand(output[2]:size(1), output[2]:size(2))

gradInput = module:backward(input, gradOutput)

end_cpu = os.time()
print(os.difftime(end_cpu, start_cpu))

