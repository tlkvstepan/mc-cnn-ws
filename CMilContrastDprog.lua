
--[[
    As input module receives 3 distance matrices: _refPos_ _refNeg_ and _posNeg_
    This module outputs two matrices nb_matches x 2 tensor.
    In first column of this tensor are energies of dprog matches for _refPos_
    In second column of this tensor are energies of dprog matches for _refNeg_
    
    Note that we contrast refPos with refNeg as well as refPos with negPos
--]]

local milContrastDprog, parent = torch.class('nn.milContrastDprog', 'nn.Module')

function milContrastDprog:__init(distMin, occ_th)
   parent.__init(self)
   
   self.occ_th = occ_th
   self.distMin = distMin
   
   --- ref-pos dprog match
   self.cols = torch.Tensor() 
   self.rows = torch.Tensor() 
   
--   -- ref-pos 2d max row-wise match
--   self.max2ColRefPos = torch.Tensor() 
   
--   -- ref-pos 2d max col-wise match
--   self.max2RowRefPos = torch.Tensor() 
   
--   --- ref-neg row-wise max match
--   self.maxRowRefNeg= torch.Tensor()
   
--   --- neg-pos col-wise max match
--   self.maxColNegPos = torch.Tensor()

end

function milContrastDprog:updateOutput(input)
  
  local E_refPos, E_refNeg, E_negPos = unpack(input)
  
  E_refPos = E_refPos:float()
  E_refNeg = E_refNeg:float()
  E_negPos = E_negPos:float()
  local E_refPos_masked = E_refPos:clone()
  
  -- allocate only once
  self.max2InRowE = self.max2InRowE or  torch.FloatTensor()
  self.max2InColE = self.max2InColE or  torch.FloatTensor()    
  self.max2InRowI  = self.max2InRowI  or torch.FloatTensor()    
  self.max2InColI = self.max2InColI or torch.FloatTensor()
    
  self.maxInNegPosColI = self.maxInNegPosColI or torch.FloatTensor()
  self.maxInNegPosColE = self.maxInNegPosColE or torch.FloatTensor()
  self.maxInRefNegRowI = self.maxInRefNegRowI or torch.FloatTensor()
  self.maxInRefNegRowE = self.maxInRefNegRowE or torch.FloatTensor()
  
  self.path_refPos = self.path_refPos or E_refPos:clone()
  self.pathNonOcc_refPos = self.pathNonOcc_refPos or E_refPos:clone()
  
  self.path_refPos:zero()
  self.pathNonOcc_refPos:zero()
      
  self.aE  = self.aE  or E_refPos:clone()
  self.aS = self.aS or E_refPos:clone()
  self.traceBack  = self.traceBack  or E_refPos:clone()
  self.aE:zero()
  self.aS:zero()
  self.traceBack:zero()
  
  t = os.clock()
  dprog.compute(E_refPos, self.path_refPos, self.aE, self.aS, self.traceBack)
  print(string.format("elapsed time: %.2f\n", os.clock() - t))
  
  dprog.findNonoccPath(self.path_refPos, self.pathNonOcc_refPos, self.occ_th)
  dprog.maskE(self.pathNonOcc_refPos, E_refPos_masked, self.distMin)

  -- for nonoccluded ref rows find max in refNeg
  local indices = self.pathNonOcc_refPos:nonzero() -- valid matches
  
  if indices:numel() > 0 then 

    self.rows = indices:select(2,1):float():add(-1) -- C++ style
    self.cols = indices:select(2,2):float():add(-1)
    local dprogE = E_refPos[self.pathNonOcc_refPos:byte()]

    -- mil forward
    self.maxInRefNegRowE:resizeAs(dprogE);
    self.maxInRefNegRowI:resizeAs(dprogE)
    dprog.findMaxForRows(E_refNeg, self.rows, self.maxInRefNegRowI, self.maxInRefNegRowE)

    -- mil backward
    self.maxInNegPosColE:resizeAs(dprogE);
    self.maxInNegPosColI:resizeAs(dprogE)
    dprog.findMaxForCols(E_negPos, self.cols, self.maxInNegPosColI, self.maxInNegPosColE)
    
    --- contrast forward
    self.max2InRowE:resizeAs(dprogE)
    self.max2InRowI:resizeAs(dprogE)
    dprog.findMaxForRows(E_refPos_masked, self.rows, self.max2InRowI, self.max2InRowE)
    
    -- contrast backward
    self.max2InColE:resizeAs(dprogE)
    self.max2InColI:resizeAs(dprogE)
    dprog.findMaxForCols(E_refPos_masked, self.cols, self.max2InColI, self.max2InColE)

    -- if cuda is on than transfer all to cuda 
    if input[1]:type() == "torch.CudaTensor" then

      self.output = {{dprogE:cuda(), self.maxInRefNegRowE:cuda()}, 
                     {dprogE:cuda(), self.maxInNegPosColE:cuda()},
                     {dprogE:cuda(), self.max2InRowE:cuda()}, 
                     {dprogE:cuda(), self.max2InColE:cuda()}}

    else

      self.output = {{dprogE:double(), self.maxInRefNegRowE:double()}, 
                     {dprogE:double(), self.maxInNegPosColE:double()},
                     {dprogE:double(), self.max2InRowE:double()},
                     {dprogE:double(), self.max2InColE:double()}}

    end

  else
    
    self.output = {}
      
  end
      
  -- note:
  -- 1. sometimes fwd or bwd can be empty
  -- 2. number of elements in fwd and bwd can be differenet 
  
  return self.output
    
end

function milContrastDprog:updateGradInput(input, gradOutput)
   
  local fwdMil, bwdMil, fwdContrast, bwdContrast = unpack(gradOutput)
  local dprogE_grad1, maxInRefNegRowE_grad = unpack(fwdMil)
  local dprogE_grad2, maxInNegPosColE_grad = unpack(bwdMil)
  local dprogE_grad3, max2InRowE_grad = unpack(fwdContrast)
  local dprogE_grad4, max2InColE_grad = unpack(bwdContrast)  
  
  local E_refPos, E_refNeg, E_negPos = unpack(input)
    
   
  self.distNegPos = self.distNegPos or E_negPos:float()
  self.distRefPos = self.distRefPos or E_refPos:float()
  self.distRefNeg = self.distRefNeg or E_refNeg:float()
  self.distNegPos:zero()
  self.distRefPos:zero()
  self.distRefNeg:zero()
  
  -- mil frw
  dprog.collect(self.distRefNeg, maxInRefNegRowE_grad:float(), self.maxInRefNegRowI, self.rows)
  dprog.collect(self.distRefPos, dprogE_grad1:float(), self.cols, self.rows)
  -- mil bwd
  dprog.collect(self.distRefPos, dprogE_grad2:float(), self.cols, self.rows)
  dprog.collect(self.distNegPos, maxInNegPosColE_grad:float(), self.cols, self.maxInNegPosColI)
  -- contrast fwd
   dprog.collect(self.distRefPos, dprogE_grad3:float(), self.cols, self.rows)
   dprog.collect(self.distRefPos, max2InRowE_grad:float(), self.max2InRowI, self.rows)
  -- contrast bwd  
  dprog.collect(self.distRefPos, dprogE_grad4:float(), self.cols, self.rows)
  dprog.collect(self.distRefPos, max2InColE_grad:float(), self.cols, self.max2InColI)
    
  if input[1]:type() == "torch.CudaTensor" then 
      self.gradInput = {self.distRefPos:cuda(), self.distRefNeg:cuda(), self.distNegPos:cuda()}
  else
      self.gradInput = {self.distRefPos:double(), self.distRefNeg:double(), self.distNegPos:double()}
  end
   
  return self.gradInput
      
end
