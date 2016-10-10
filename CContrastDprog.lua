
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
  
  local dim = input:size(1)

  
  
  -- dprog (always on cpu)
  self.path = self.path or input:float()     
  self.path:zero()
  self.E = self.E or input:float()
  self.E:copy(input)
  self.aE  = self.aE  or input:float()    
  self.aS = self.aS or input:float()
  self.traceBack  = self.traceBack  or input:float()
  self.aE:zero()
  self.aS:zero()
  self.traceBack:zero()
  dprog.compute(self.E, self.path,  self.aE, self.aS, self.traceBack)
  
  -- mask occluded
  self.pathNonOcc = self.pathNonOcc or input:clone() -- cuda if cuda mode
  self.pathNonOcc:copy(self.path)
  
  local mask = torch.repeatTensor(self.pathNonOcc:sum(2):gt(self.occTh), 1, dim)
  mask:add(torch.repeatTensor(self.pathNonOcc:sum(1):gt(self.occTh), dim, 1))
  self.pathNonOcc[mask] = 0;
  local dprogE = input[self.pathNonOcc:byte()]
  
  local E_masked = input -- cuda if cuda mode
  local E_masked_VEC = E_masked:view(dim*dim)
  local indices = self.pathNonOcc:float():nonzero() -- cuda not supported for this opperation
  if( input:type() == "torch.CudaTensor"  )then
    indices = indices:cuda()
  end
  
  -- if there are nonoccluded path segments
  if indices:numel() > 0 then
    
    self.rows = indices:select(2,1)
    self.cols = indices:select(2,2)
    
    -- mask energy array
    for dy = -self.distMin,self.distMin do
      local rowsMask = self.rows + dy;
      rowsMask[rowsMask:gt(dim)] = dim;
      rowsMask[rowsMask:lt(1)] = 1;
      for dx = -self.distMin,self.distMin do
        local colsMask = self.cols + dx;
        colsMask[colsMask:gt(dim)] = dim;
        colsMask[colsMask:lt(1)] = 1;
        local idx = colsMask + (rowsMask-1)*dim
        E_masked_VEC:indexFill(1, idx, -1/0)
      end
    end

    -- compute maximum
    self.rowwiseMaxE, self.rowwiseMaxI = E_masked:max(2)
    self.rowwiseMaxE = self.rowwiseMaxE:index(1,self.rows):squeeze()
    self.rowwiseMaxI = self.rowwiseMaxI:index(1,self.rows):squeeze()
    self.colwiseMaxE, self.colwiseMaxI = E_masked:max(1)
    self.colwiseMaxE = self.colwiseMaxE:index(2,self.cols):squeeze()
    self.colwiseMaxI = self.colwiseMaxI:index(2,self.cols):squeeze()
    
    self.output = {{dprogE, self.rowwiseMaxE}, {dprogE, self.colwiseMaxE}}
  
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
   self.gradInput = self.gradInput:resizeAs(input):zero() -- same type as input
   local dim = input:size(1);
   local gradInput_vec = self.gradInput:view(dim*dim) 
   local idx;
   
   idx = (self.cols) + (self.rows-1)*dim;
   gradInput_vec:indexAdd(1, idx, gradOutput_dprog1)
   gradInput_vec:indexAdd(1, idx, gradOutput_dprog2)
  
   idx = (self.rowwiseMaxI) + (self.rows-1)*dim;
   gradInput_vec:indexAdd(1, idx, gradOutput_row)
   
   idx = (self.cols) + (self.colwiseMaxI-1)*dim;
   gradInput_vec:indexAdd(1, idx, gradOutput_col)
   
   return self.gradInput
end
