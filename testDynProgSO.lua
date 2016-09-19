require 'libdynprog'
require 'gnuplot'

local input = torch.rand(30,40):float()
local aE =  torch.FloatTensor(30,40)
local aP = torch.FloatTensor(30,40)

indices = torch.Tensor(30):float()
values = torch.Tensor(30):float()

dynprog.compute(input, aE, aP, indices, values);

print(values)
print(indices)
