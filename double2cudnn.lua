require 'nn'
require 'cunn'
require 'cudnn'


mod = nn.SpatialConvolution(3, 3, 3, 3, 1, 1);
mod_cudnn = nn.SpatialConvolution(3, 3, 3, 3, 1, 1);
mod_cudnn:cuda()
cudnn.convert(mod_cudnn, cudnn)
 
parm_cudnn = mod_cudnn:getParameters() 
parm_double = mod:getParameters() 
parm_double:copy(parm_cudnn)


mod_clone = mod_cudnn:clone():double()
parm_clone = mod_clone:getParameters() 

print(parm_cudnn[4])
print(parm_double[4])
print(parm_clone[4])

