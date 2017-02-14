
--[[
    As input module receives 3 distance matrices: _refPos_ _refNeg_ and _posNeg_
    This module outputs two matrices nb_matches x 2 tensor.
    In first column of this tensor are energies of dprog matches for _refPos_
    In second column of this tensor are energies of dprog matches for _refNeg_
    
    Note that we contrast refPos with refNeg as well as refPos with negPos
--]]

local milContrastive, parent = torch.class('nn.milContrastive', 'nn.Module')

function milContrastive:__init(th_sup, th_occ, disp_max)
   parent.__init(self)
   
   self.th_sup = th_sup
   self.th_occ = th_occ
   self.disp_max = disp_max
   
   self.I_maxInRow_refPos = torch.Tensor() 
   self.I_maxInRow_refNeg = torch.Tensor() 

   self.I_maxInCol_refPos = torch.Tensor() 
   self.I_maxInCol_negPos = torch.Tensor() 

   self.I_2maxInRow_refPos = torch.Tensor()
   self.I_2maxInCol_refPos = torch.Tensor()

end

function milContrastive:updateOutput(input)
  
  local E_refPos, E_refNeg, E_negPos = unpack(input)
  local E_refPosFwd = E_refPos; 
  local E_refPosBwd = E_refPos:clone();
  
  local dim = E_refPos:size(1)
  local outdim = dim - (1 + self.disp_max) + 1; 
    
  -- milfwd
  local E_maxInRow_refPos, I_maxInRow_refPos = torch.max(E_refPos, 2)
  local E_maxInRow_refNeg, I_maxInRow_refNeg = torch.max(E_refNeg, 2)
  
  -- milbwd
  local E_maxInCol_refPos, I_maxInCol_refPos = torch.max(E_refPos, 1) 
  local E_maxInCol_negPos, I_maxInCol_negPos = torch.max(E_negPos, 1) 
   
    
  -- contrastive fwd-bwd
  -- mask maximum and all neighbours of the max
  for d = -self.th_sup, self.th_sup do
    
     -- fwd
     local ind =  I_maxInRow_refPos + d
     ind[ind:lt(1)] = 1
     ind[ind:gt(dim)] = dim
     E_refPosFwd = E_refPosFwd:scatter(2, ind, -1/0)
  
     -- bwd 
     ind =  I_maxInCol_refPos + d
     ind[ind:lt(1)] = 1
     ind[ind:gt(dim)] = dim
     E_refPosBwd = E_refPosBwd:scatter(1, ind, -1/0)
  
  end


  E_maxInRow_refPos:view(E_maxInRow_refPos, dim)
  E_maxInRow_refNeg:view(E_maxInRow_refNeg, dim)
  I_maxInRow_refPos:view(I_maxInRow_refPos, dim)
  I_maxInRow_refNeg:view(I_maxInRow_refNeg, dim)
  --
  E_maxInCol_refPos:view(E_maxInCol_refPos, dim)
  E_maxInCol_negPos:view(E_maxInCol_negPos, dim)
  I_maxInCol_refPos:view(I_maxInCol_refPos, dim)
  I_maxInCol_negPos:view(I_maxInCol_negPos, dim)
  
  local E_2maxInRow_refPos, I_2maxInRow_refPos = torch.max(E_refPosFwd, 2)
  local E_2maxInCol_refPos, I_2maxInCol_refPos = torch.max(E_refPosBwd, 1)
  
  E_2maxInRow_refPos:view(E_2maxInRow_refPos, dim)
  I_2maxInRow_refPos:view(I_2maxInRow_refPos, dim)
  --
  E_2maxInCol_refPos:view(E_2maxInCol_refPos, dim)
  I_2maxInCol_refPos:view(I_2maxInCol_refPos, dim)
    
  -- cut top disp_max rows
  E_maxInRow_refPos = E_maxInRow_refPos[{{1+self.disp_max, dim}}]
  E_maxInRow_refNeg = E_maxInRow_refNeg[{{1+self.disp_max, dim}}]
  E_2maxInRow_refPos = E_2maxInRow_refPos[{{1+self.disp_max, dim}}]
  --
  self.I_maxInRow_refPos = I_maxInRow_refPos[{{1+self.disp_max, dim}}]:cuda() 
  self.I_maxInRow_refNeg = I_maxInRow_refNeg[{{1+self.disp_max, dim}}]:cuda()
  self.I_2maxInRow_refPos = I_2maxInRow_refPos[{{1+self.disp_max, dim}}]:cuda()
   
  -- cut last disp_max cols
  E_maxInCol_refPos = E_maxInCol_refPos[{{1, dim-self.disp_max}}] 
  E_maxInCol_negPos = E_maxInCol_negPos[{{1, dim-self.disp_max}}]
  E_2maxInCol_refPos = E_2maxInCol_refPos[{{1, dim-self.disp_max}}]
  --
  self.I_maxInCol_refPos = I_maxInCol_refPos[{{1, dim-self.disp_max}}]:cuda()
  self.I_maxInCol_negPos = I_maxInCol_negPos[{{1, dim-self.disp_max}}]:cuda()
  self.I_2maxInCol_refPos= I_2maxInCol_refPos[{{1, dim-self.disp_max}}]:cuda() 
  
  self.output = {{E_maxInRow_refPos, E_maxInRow_refNeg},  -- mil fwd 
                 {E_maxInCol_refPos, E_maxInCol_negPos},  -- mil bwd, every tensor is width-disp_max-hpatch*2
                 {E_maxInRow_refPos, E_2maxInRow_refPos}, -- fwdContrastive
                 {E_maxInCol_refPos, E_2maxInCol_refPos}} -- bwdContrastive
                   
  return self.output 
  
end

function milContrastive:updateGradInput(input, gradOutput)
   
  local fwdMil, bwdMil, fwdContrast, bwdContrast = unpack(gradOutput)
 
  local ograd_maxInRow_refPos1, ograd_maxInRow_refNeg = unpack(fwdMil)
  local ograd_maxInCol_refPos1, ograd_maxInCol_negPos = unpack(bwdMil)
  local ograd_maxInRow_refPos2, ograd_2maxInRow_refPos =  unpack(fwdContrast)
  local ograd_maxInCol_refPos2, ograd_2maxInCol_refPos =  unpack(fwdContrast)
  
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
  
  local row = torch.range(1+self.disp_max, dim):cuda()
  local col = torch.range(1, dim - self.disp_max):cuda()
    
  idx = (self.I_maxInRow_refPos) + (row-1)*dim;
  igradRefPos_vec:indexAdd(1, idx, ograd_maxInRow_refPos1)
  igradRefPos_vec:indexAdd(1, idx, ograd_maxInRow_refPos2)
  
  idx = col + (self.I_maxInCol_refPos-1)*dim;
  igradRefPos_vec:indexAdd(1, idx, ograd_maxInCol_refPos1)
  igradRefPos_vec:indexAdd(1, idx, ograd_maxInCol_refPos2)
    
  -- mil fwd
  idx = (self.I_maxInRow_refNeg) + (row-1)*dim;
  igradRegNeg_vec:indexAdd(1, idx, ograd_maxInRow_refNeg)
  
  -- mil bwd
  idx = (col) + (self.I_maxInCol_negPos-1)*dim;
  igradNegPos_vec:indexAdd(1, idx, ograd_maxInCol_negPos)
  
  -- contrast fwd
  idx = (self.I_2maxInRow_refPos) + (row-1)*dim;
  igradRefPos_vec:indexAdd(1, idx, ograd_2maxInRow_refPos)
    
  -- contrast bwd  
  idx = (col) + (self.I_2maxInCol_refPos-1)*dim;
  igradRefPos_vec:indexAdd(1, idx, ograd_2maxInCol_refPos)
  
  self.gradInput = {self.igradRefPos, self.igradRefNeg, self.igradNegPos}
   
  return self.gradInput
      
end
