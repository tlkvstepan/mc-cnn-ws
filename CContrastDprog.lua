
--[[
  Given h x w tensor as an _input_, module outputs 4 x h tensor
    
    first column  - fDprog
    second column - fMax
    third column  - bDprog
    forth column  - bMax

--]]

local contrastDprog, parent = torch.class('nn.contrastDprog', 'nn.Module')

function contrastDprog:__init(distMin, occTh)
   parent.__init(self)
   self.distMin = distMin
   self.occTh = occTh 
   -- these vector store indices of for Dyn Prog solution and row-wise maximums
   self.cols = torch.Tensor()
   self.rows = torch.Tensor()
   --self.rowwiseMaxI = torch.Tensor()
   --self.colwiseMaxI = torch.Tensor()
 
end

function contrastDprog:updateOutput(input)
  
  local E = input:float()
  local E_masked = E:clone()
  
  -- allocate local arrays only once
  self.rowwiseMaxE = self.rowwiseMaxE or  torch.FloatTensor()
  self.colwiseMaxE = self.colwiseMaxE or  torch.FloatTensor()    
  self.rowwiseMaxI  = self.rowwiseMaxI  or torch.FloatTensor()    
  self.colwiseMaxI = self.colwiseMaxI or torch.FloatTensor()
  
  self.path = self.path or input:clone():float()     
  self.path:zero()
  
  self.pathNonOcc = self.pathNonOcc or input:clone():float()
  self.pathNonOcc:zero()
      
  dprog.compute(E, self.path)
  dprog.findNonoccPath(self.path, self.pathNonOcc, self.occTh)
  dprog.maskE(self.pathNonOcc, E_masked, self.distMin)

  local indices = self.pathNonOcc:nonzero() -- valid matches
  
  if indices:numel() > 0 then

    self.rows = indices:select(2,1):float():add(-1) -- C++ style
    self.cols = indices:select(2,2):float():add(-1)
    
    local dprogE = E[self.pathNonOcc:byte()]
    local dim = dprogE:numel()
    self.rowwiseMaxI:resizeAs(dprogE)
    self.rowwiseMaxE:resizeAs(dprogE)
    self.colwiseMaxI:resizeAs(dprogE)
    self.colwiseMaxE:resizeAs(dprogE)
    dprog.findMaxForRows(E_masked, self.rows, self.rowwiseMaxI, self.rowwiseMaxE)
    dprog.findMaxForCols(E_masked, self.cols, self.colwiseMaxI, self.colwiseMaxE)

    -- if cuda is on than transfer all to cuda 
    if input:type() == "torch.CudaTensor" then
      
      self.output = {{dprogE:cuda(), self.rowwiseMaxE:cuda()}, {dprogE:cuda(), self.colwiseMaxE:cuda()}}
  
    else

      self.output = {{dprogE:double(), self.rowwiseMaxE:double()}, {dprogE:double(), self.colwiseMaxE:double()}}
  
    end
    
  else
   
    self.output = {}
  
  end
  
  return self.output
end

function contrastDprog:updateGradInput(input, gradOutput)
      
   local fwd, bwd  = unpack(gradOutput)
   local gradOutput_dprog1, gradOutput_row = unpack(fwd)
   local gradOutput_dprog2, gradOutput_col = unpack(bwd)
   
   -- pass input gradient to dyn prog and max 
   self.gradInput = self.gradInput:resizeAs(input):zero():float()
   
   dprog.collect(self.gradInput, gradOutput_dprog1:float(), self.cols, self.rows)
   dprog.collect(self.gradInput, gradOutput_row:float(), self.rowwiseMaxI, self.rows)
   
   dprog.collect(self.gradInput, gradOutput_dprog2:float(), self.cols, self.rows)
   dprog.collect(self.gradInput, gradOutput_col:float(), self.cols, self.colwiseMaxI)
    
   if input:type() == "torch.CudaTensor" then 
    self.gradInput = self.gradInput:cuda()
   else
    self.gradInput = self.gradInput:double() 
   end
      
   return self.gradInput
end
