
--[[
    As input module receives 3 distance matrices: _refPos_ _refNeg_ and _posNeg_
    This module outputs two matrices nb_matches x 2 tensor.
    In first column of this tensor are energies of dprog matches for _refPos_
    In second column of this tensor are energies of dprog matches for _refNeg_
    
    Note that we contrast refPos with refNeg as well as refPos with negPos
--]]

local milDprog, parent = torch.class('nn.milDprog', 'nn.Module')

function milDprog:__init(occ_th)
   parent.__init(self)
   
   self.occTh = occ_th
   
   self.cols = torch.Tensor()
   self.rows = torch.Tensor()
 
end

function milDprog:updateOutput(input)
  
  local E_refPos, E_refNeg, E_negPos = unpack(input)
  
  local dim = E_refPos:size(1)
  local gpu_on = (E_refPos:type() == "torch.CudaTensor")

  -- dprog (on cpu) 
  self.path_refPos = self.path_refPos or E_refPos:float()     
  self.path_refPos:zero()
  self.E_refPos = self.E_refPos or E_refPos:float()
  self.E_refPos:copy(E_refPos) -- copy input GPU to CPU if cuda mode
  self.aE  = self.aE  or E_refPos:float()    
  self.aS = self.aS or E_refPos:float()
  self.traceBack  = self.traceBack  or E_refPos:float()
  self.aE:zero()
  self.aS:zero()
  self.traceBack:zero()
  dprog.compute(self.E_refPos, self.path_refPos, self.aE, self.aS, self.traceBack)
  
  -- mask occluded (on GPU if avaliable)
  self.pathNonOcc_refPos = self.pathNonOcc_refPos or E_refPos:clone() -- cuda if cuda mode
  self.pathNonOcc_refPos:copy(self.path_refPos)
  local mask = torch.repeatTensor(self.pathNonOcc_refPos:sum(2):gt(self.occTh), 1, dim)
  mask:add(torch.repeatTensor(self.pathNonOcc_refPos:sum(1):gt(self.occTh), dim, 1))
  self.pathNonOcc_refPos[mask] = 0;
  local dprogE = E_refPos[self.pathNonOcc_refPos:byte()]
  
  -- find nooccluded rows / cols 
  local indices = self.pathNonOcc_refPos:float():nonzero() -- cuda not supported for this opperation
  
  -- if there are nonoccluded segments
  if( indices:numel() > 0 ) then

    if( gpu_on ) then
      
      indices = indices:cuda()
    
    end
    
    self.rows = indices:select(2,1)
    self.cols = indices:select(2,2)
      
    -- find max matches in refNeg and posNeg for nonoccluded 
    -- rows / cols
    self.maxRefNegE, self.maxRefNegI = E_refNeg:max(2)
    self.maxRefNegE = self.maxRefNegE:index(1,self.rows):squeeze()
    self.maxRefNegI = self.maxRefNegI:index(1,self.rows):squeeze()
    
    self.maxNegPosE, self.maxNegPosI = E_negPos:max(1)
    self.maxNegPosE = self.maxNegPosE:index(2,self.cols):squeeze()
    self.maxNegPosI = self.maxNegPosI:index(2,self.cols):squeeze()
  
    self.output = {{dprogE, self.maxRefNegE}, {dprogE, self.maxNegPosE}}
  
  else 
  
   self.output = {}
    
  end
  
  return self.output
    
end

function milDprog:updateGradInput(input, gradOutput)
   
  local fwd, bwd = unpack(gradOutput)
  local ogradRefPos, ogradRefNeg = unpack(fwd)
  local ogradPosRef, ogradNegPos = unpack(bwd)
  
  local E_refPos, E_refNeg, E_negPos = unpack(input)
  dim = E_refPos:size(1)  
--  local dprogPosRefE_grad = dprogPosRefE_grad:float()
--  local dprogNegPosE_grad = dprogNegPosE_grad:float()
  
--  local dprogRefPosE_grad = dprogRefPosE_grad:float()
--  local dprogRefNegE_grad = dprogRefNegE_grad:float()  
  
  self.igradNegPos = self.gradNegPos or E_negPos:clone()
  self.igradRefPos = self.gradRefPos or E_refPos:clone()
  self.igradRefNeg = self.gradRefNeg or E_refNeg:clone()
  self.igradNegPos:zero()
  self.igradRefPos:zero()
  self.igradRefNeg:zero()
  
  local igradRefPos_vec = self.igradRefPos:view(dim*dim) 
  local igradRegNeg_vec = self.igradRefNeg:view(dim*dim) 
  local igradNegPos_vec = self.igradNegPos:view(dim*dim) 
  local idx;
      
  idx = (self.cols) + (self.rows-1)*dim;
  igradRefPos_vec:indexAdd(1, idx, ogradRefPos)
  igradRefPos_vec:indexAdd(1, idx, ogradPosRef)
  
  idx = (self.maxRefNegI) + (self.rows-1)*dim;
  igradRegNeg_vec:indexAdd(1, idx, ogradRefNeg)
   
  idx = (self.cols) + (self.maxNegPosI-1)*dim;
  igradNegPos_vec:indexAdd(1, idx, ogradNegPos)
  
  self.gradInput = {igradRefPos_vec, igradRegNeg_vec, igradNegPos_vec}
   
  return self.gradInput
end
