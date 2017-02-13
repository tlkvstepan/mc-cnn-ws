-- Dealing with the CPU/GPU
-- mynn will take entries in that order: mynn, cudnn, cunn, nn
mynn = {}

local mt = {}

function mt.__index(table, key)
   return (cudnn and cudnn[key]) or (cunn and cunn[key]) or nn[key]
end

setmetatable(mynn, mt)

-- These are the tensors that can be kept on the CPU
mynn.SlowTensor = torch.Tensor
-- These are the tensors that should be moved to the GPU
mynn.FastTensor = torch.CudaTensor