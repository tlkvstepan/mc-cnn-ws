
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
   
   self.occ_th = occ_th
   -- self.cols 
   -- self.rows 
   ---
   -- self.matchColRefNeg 
   ----
   -- self.matchRowNegPos 

end

function milDprog:updateOutput(input)
  
  local E_refPos, E_refNeg, E_negPos = unpack(input)
  
  E_refPos = E_refPos:float()
  E_refNeg = E_refNeg:float()
  E_negPos= E_negPos:float()
  
  -- allocate only once
  self.matchRowNegPos = self.matchRowNegPos or torch.FloatTensor()
  self.maxNegPosE = self.maxNegPosE or torch.FloatTensor()
  self.matchColRefNeg = self.matchColRefNeg or torch.FloatTensor()
  self.maxRefNegE = self.maxRefNegE or torch.FloatTensor()
  self.path_refPos = self.path_refPos or E_refPos:clone()
  self.pathNonOcc_refPos = self.pathNonOcc_refPos or E_refPos:clone()
  self.path_refPos:zero()
  self.pathNonOcc_refPos:zero()
      
  self.aE  = self.aE  or input:clone():float()    
  self.aS = self.aS or input:clone():float()
  self.traceBack  = self.traceBack  or input:clone():float()
  self.aE:zero()
  self.aS:zero()
  self.traceBack:zero()
  
  dprog.compute(E_refPos, self.path_refPos, self.aE, self.aS, self.traceBack)
  dprog.findNonoccPath(self.path_refPos, self.pathNonOcc_refPos, self.occ_th)
  
  -- for nonoccluded ref rows find max in refNeg
  local indices = self.pathNonOcc_refPos:nonzero() -- valid matches
  
  if   indices:numel() > 0 then 

    self.rows = indices:select(2,1):float():add(-1) -- C++ style
    self.cols = indices:select(2,2):float():add(-1)
    local dprogE = E_refPos[self.pathNonOcc_refPos:byte()]

    self.maxRefNegE:resizeAs(dprogE);
    self.matchColRefNeg:resizeAs(dprogE)
    dprog.findMaxForRows(E_refNeg, self.rows, self.matchColRefNeg, self.maxRefNegE)

    self.maxNegPosE:resizeAs(dprogE);
    self.matchRowNegPos:resizeAs(dprogE)
    dprog.findMaxForCols(E_negPos, self.cols, self.matchRowNegPos, self.maxNegPosE)

    -- if cuda is on than transfer all to cuda 
    if input[1]:type() == "torch.CudaTensor" then

      self.output = {{dprogE:cuda(), self.maxRefNegE:cuda()}, {dprogE:cuda(), self.maxNegPosE:cuda()}}

    else

      self.output = {{dprogE:double(), self.maxRefNegE:double()}, {dprogE:double(), self.maxNegPosE:double()}}

    end

  else
    
    self.output = {}
      
  end
      
  -- note:
  -- 1. sometimes fwd or bwd can be empty
  -- 2. number of elements in fwd and bwd can be differenet 
  
  return self.output
    
end

function milDprog:updateGradInput(input, gradOutput)
   
  local fwd, bwd = unpack(gradOutput)
  local dprogRefPosE_grad, dprogRefNegE_grad = unpack(fwd)
  local dprogPosRefE_grad, dprogNegPosE_grad = unpack(bwd)
  
  local E_refPos, E_refNeg, E_negPos = unpack(input)
    
  local dprogPosRefE_grad = dprogPosRefE_grad:float()
  local dprogNegPosE_grad = dprogNegPosE_grad:float()
  
  local dprogRefPosE_grad = dprogRefPosE_grad:float()
  local dprogRefNegE_grad = dprogRefNegE_grad:float()  
  
  self.distNegPos = self.distNegPos or E_negPos:float()
  self.distRefPos = self.distRefPos or E_refPos:float()
  self.distRefNeg = self.distRefNeg or E_refNeg:float()
  self.distNegPos:zero()
  self.distRefPos:zero()
  self.distRefNeg:zero()
  
  
  dprog.collect(self.distRefNeg, dprogRefNegE_grad, self.matchColRefNeg, self.rows)
  
  dprog.collect(self.distRefPos, dprogRefPosE_grad, self.cols, self.rows)
  dprog.collect(self.distRefPos, dprogPosRefE_grad, self.cols, self.rows)
  
  dprog.collect(self.distNegPos, dprogNegPosE_grad, self.cols, self.matchRowNegPos)
   
  
  if input[1]:type() == "torch.CudaTensor" then 
      self.gradInput = {self.distRefPos:cuda(), self.distRefNeg:cuda(), self.distNegPos:cuda()}
  else
      self.gradInput = {self.distRefPos:double(), self.distRefNeg:double(), self.distNegPos:double()}
  end
   
  return self.gradInput
end
