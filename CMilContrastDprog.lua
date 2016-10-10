
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
   
   self.occTh = occ_th
   self.distMin = distMin
   
   --- ref-pos dprog match
   self.cols = torch.Tensor() 
   self.rows = torch.Tensor() 

end

function milContrastDprog:updateOutput(input)
  
  local E_refPos, E_refNeg, E_negPos = unpack(input)
  local dim = E_refPos:size(1)
  local gpu_on = E_refPos:type() == "torch.CudaTensor"
  
  -- dprog (always on cpu)
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
  local dprogE = E_refPos[self.pathNonOcc_refPos:byte()] -- this basically copy
  
  -- find nooccluded rows / cols 
  local indices = self.pathNonOcc_refPos:float():nonzero() -- cuda not supported for this opperation
  
  -- if there are nonoccluded segments
  if( indices:numel() > 0 ) then
    
    if( gpu_on ) then
      
      indices = indices:cuda()
    
    end
    
    self.rows = indices:select(2,1)
    self.cols = indices:select(2,2)
    
    -- mask energy array
    local E_refPos_masked = E_refPos -- cuda if cuda mode
    local E_refPos_masked_vec = E_refPos_masked:view(dim*dim)
    for dy = -self.distMin, self.distMin do
      
      local rowsMask = self.rows + dy;
      rowsMask[rowsMask:gt(dim)] = dim;
      rowsMask[rowsMask:lt(1)] = 1;
      
      for dx = -self.distMin,self.distMin do
      
        local colsMask = self.cols + dx;
        colsMask[colsMask:gt(dim)] = dim;
        colsMask[colsMask:lt(1)] = 1;
        local idx = colsMask + (rowsMask-1)*dim
        E_refPos_masked_vec:indexFill(1, idx, -1/0)
     
      end
    end
    
    -- mil forward
    self.maxInRefNegRowE, self.maxInRefNegRowI = E_refNeg:max(2)
    self.maxInRefNegRowE = self.maxInRefNegRowE:index(1,self.rows):squeeze()
    self.maxInRefNegRowI = self.maxInRefNegRowI:index(1,self.rows):squeeze()
    
    -- mil backward
    self.maxInNegPosColE, self.maxInNegPosColI = E_negPos:max(1)
    self.maxInNegPosColE = self.maxInNegPosColE:index(2,self.cols):squeeze()
    self.maxInNegPosColI = self.maxInNegPosColI:index(2,self.cols):squeeze()
    
    --- contrast forward
    self.max2InRowE, self.max2InRowI = E_refPos_masked:max(2)
    self.max2InRowE = self.max2InRowE:index(1,self.rows):squeeze()
    self.max2InRowI = self.max2InRowI:index(1,self.rows):squeeze()
    
    -- contrast backward
    self.max2InColE, self.max2InColI = E_refPos_masked:max(1)
    self.max2InColE = self.max2InColE:index(2,self.cols):squeeze()
    self.max2InColI = self.max2InColI:index(2,self.cols):squeeze()
    

    self.output = {{dprogE, self.maxInRefNegRowE},  -- mil fwd 
                   {dprogE, self.maxInNegPosColE},  -- mil bwd
                   {dprogE, self.max2InRowE}, -- contrast fwd
                   {dprogE, self.max2InColE}} -- contrast bwd
    
  else
    
    self.output = {}
      
  end
      
  -- note:
  -- 1. sometimes output is empty
  -- 2. number of elements in fwd and bwd can be differenet 
  
  return self.output
    
end

function milContrastDprog:updateGradInput(input, gradOutput)
   
  local fwdMil, bwdMil, fwdContrast, bwdContrast = unpack(gradOutput)
 
  local ogradDprogE1, ogradMaxInRefNegRowE = unpack(fwdMil)
  local ogradDprogE2, ogradMaxInNegPosColE = unpack(bwdMil)
  local ogradDprogE3, ogradMax2InRowE = unpack(fwdContrast)
  local ogradDprogE4, ogradMax2InColE = unpack(bwdContrast)  
  
  local E_refPos, E_refNeg, E_negPos = unpack(input)
  local dim = E_refPos:size(1) 
  
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
  igradRefPos_vec:indexAdd(1, idx, ogradDprogE1)
  igradRefPos_vec:indexAdd(1, idx, ogradDprogE2)
  igradRefPos_vec:indexAdd(1, idx, ogradDprogE3)
  igradRefPos_vec:indexAdd(1, idx, ogradDprogE4)
  
  -- mil fwd
  idx = (self.maxInRefNegRowI) + (self.rows-1)*dim;
  igradRegNeg_vec:indexAdd(1, idx, ogradMaxInRefNegRowE)
  
  -- mil bwd
  idx = (self.cols) + (self.maxInNegPosColI-1)*dim;
  igradNegPos_vec:indexAdd(1, idx, ogradMaxInNegPosColE)
  
  -- contrast fwd
  idx = (self.max2InRowI) + (self.rows-1)*dim;
  igradRefPos_vec:indexAdd(1, idx, ogradMax2InRowE)
    
  -- contrast bwd  
  idx = (self.cols) + (self.max2InColI-1)*dim;
  igradRefPos_vec:indexAdd(1, idx, ogradMax2InColE)
  
  self.gradInput = {self.igradRefPos, self.igradRefNeg, self.igradNegPos}
   
  return self.gradInput
      
end
