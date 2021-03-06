
--[[
  Given h x w tensor as an _input_, module outputs 4 x h tensor
    
    first column  - fDprog
    second column - fMax
    third column  - bDprog
    forth column  - bMax

--]]

local contrastiveDP, parent = torch.class('nn.contrastiveDP', 'nn.Module')

function contrastiveDP:__init(th_sup, th_occ)
   
   parent.__init(self)
   
   self.th_sup = th_sup
   self.th_occ = th_occ
   
   -- these vector store indices of for Dyn Prog solution and row-wise maximums
   self.cols = torch.Tensor()
   self.rows = torch.Tensor()
end

function contrastiveDP:updateOutput(input)
  
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
  
  local mask = torch.repeatTensor(self.pathNonOcc:sum(2):gt(self.th_occ), 1, dim)
  mask:add(torch.repeatTensor(self.pathNonOcc:sum(1):gt(self.th_occ), dim, 1))
  self.pathNonOcc[mask] = 0;
  local dprogE = input[self.pathNonOcc:byte()]
  
  local E_masked = input -- cuda if cuda mode
  local E_masked_VEC = E_masked:view(dim*dim)
  local indices = self.pathNonOcc:double():nonzero():cuda() -- cuda not supported for this opperation
  --if( input:type() == "torch.CudaTensor"  )then
 -- local indices = torch.CudaLongTensor()
  --indices:resize(indices:size())
  --indices = indices:cuda()
  --end
  
  -- if there are nonoccluded path segments
  
  if indices:numel() > 2 then
 
    self.rows = indices:select(2,1)
    self.cols = indices:select(2,2)
 
    -- mask energy array
    local rowsMask = torch.CudaTensor()
    local colsMask = torch.CudaTensor()
    rowsMask:resize(self.rows:size())
    colsMask:resize(self.cols:size())
    for dy = -self.th_sup,self.th_sup do
      rowsMask:copy(self.rows + dy);
      rowsMask[rowsMask:gt(dim)] = dim;
      rowsMask[rowsMask:lt(1)] = 1;
      for dx = -self.th_sup,self.th_sup do
        colsMask:copy(self.cols + dx);
        colsMask[colsMask:gt(dim)] = dim;
        colsMask[colsMask:lt(1)] = 1;
        local idx = colsMask + (rowsMask-1)*dim
        E_masked_VEC:indexFill(1, idx, -1/0)
      end
    end

    -- compute maximum
    self.rowwiseMaxE, self.rowwiseMaxI = E_masked:max(2)
    self.rowwiseMaxE = self.rowwiseMaxE:index(1,self.rows:long()):squeeze():cuda()
    self.rowwiseMaxI = self.rowwiseMaxI:index(1,self.rows:long()):squeeze():cuda()
    self.colwiseMaxE, self.colwiseMaxI = E_masked:max(1)
    self.colwiseMaxE = self.colwiseMaxE:index(2,self.cols:long()):squeeze():cuda()
    self.colwiseMaxI = self.colwiseMaxI:index(2,self.cols:long()):squeeze():cuda()
    
    self.rows = self.rows:cuda()
    self.cols = self.cols:cuda() 
    
    
    self.output = {{dprogE, self.rowwiseMaxE}, {dprogE, self.colwiseMaxE}}
  
  else
   
    self.output = {}
  
  end
  
  return self.output
end

function contrastiveDP:updateGradInput(input, gradOutput)
      
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
