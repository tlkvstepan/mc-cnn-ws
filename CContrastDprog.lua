
--[[
  Given h x w tensor as an _input_, module outputs 4 x h tensor
    
    first column  - fDprog
    second column - fMax
    third column  - bDprog
    forth column  - bMax

--]]

local contrastDprog, parent = torch.class('nn.contrastDprog', 'nn.Module')

function contrastDprog:__init(distMin)
   parent.__init(self)
   self.distMin = distMin
   -- these vector store indices of for Dyn Prog solution and row-wise maximums
   self.dprogE = torch.Tensor()
   self.cols = torch.Tensor()
   self.rows = torch.Tensor()
   self.rowwiseMaxI = torch.Tensor()
   self.colwiseMaxI = torch.Tensor()
 
end

function contrastDprog:updateOutput(input)
  
  local E = input:clone():float()
  local E_masked = input:clone():float()
  local path = input:clone():zero():float()
  local pathNonOcc = input:clone():zero():float()
  local aE = input:clone():zero():float()
  local aS = input:clone():zero():float()
  
  dprog.compute(E, path, aE, aS)
  dprog.findNonoccPath(path, pathNonOcc)
  dprog.maskE(pathNonOcc, E_masked, self.distMin)

  indices = pathNonOcc:nonzero() -- valid matches
  self.rows = indices[{{},{1}}]:clone():squeeze():float():add(-1) -- C++ style
  self.cols = indices[{{},{2}}]:clone():squeeze():float():add(-1)

  dprogE = E[pathNonOcc:byte()]:clone()
  dim = dprogE:numel()
  self.rowwiseMaxI = torch.zeros(dim):float()
  rowwiseMaxE = torch.zeros(dim):float()
  self.colwiseMaxI = torch.zeros(dim):float()
  colwiseMaxE = torch.zeros(dim):float()
  dprog.findMaxForRows(E_masked, self.rows, self.rowwiseMaxI, rowwiseMaxE)
  dprog.findMaxForCols(E_masked, self.cols, self.colwiseMaxI, colwiseMaxE)

 
  dprogE = dprogE:double()
  rowwiseMaxE = rowwiseMaxE:double()
  colwiseMaxE = colwiseMaxE:double()
  
  -- if cuda is on than transfer all to cuda 
  if input:type() == "torch.CudaTensor" then
 
    dprogE = dprogE:cuda()
    rowwiseMaxE = rowwiseMaxE:cuda()
    colwiseMaxE = colwiseMaxE:cuda()
    
  end

  self.output = torch.cat({nn.utils.addSingletonDimension(dprogE,2), nn.utils.addSingletonDimension(rowwiseMaxE,2), nn.utils.addSingletonDimension(colwiseMaxE,2)}, 2)
   
  return self.output
end

function contrastDprog:updateGradInput(input, gradOutput)
   
   -- pass input gradient to dyn prog and max 
   self.gradInput = self.gradInput:resizeAs(input):zero():float()
   
   dprog.collect(self.gradInput, gradOutput:select(2,1):float(), self.cols, self.rows)
   dprog.collect(self.gradInput, gradOutput:select(2,2):float(), self.rowwiseMaxI, self.rows)
   dprog.collect(self.gradInput, gradOutput:select(2,3):float(), self.cols, self.colwiseMaxI)
    
   self.gradInput = self.gradInput:double() 
   if input:type() == "torch.CudaTensor" then 
    self.gradInput = self.gradInput:cuda()
   end
      
   return self.gradInput
end
