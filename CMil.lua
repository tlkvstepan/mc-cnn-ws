
--[[
    As input module receives 3 distance matrices: _refPos_ _refNeg_ and _posNeg_
    This module outputs two matrices nb_matches x 2 tensor.
    In first column of this tensor are energies of dprog matches for _refPos_
    In second column of this tensor are energies of dprog matches for _refNeg_
     Note that we contrast refPos with refNeg as well as refPos with negPos
--]]

local mil, parent = torch.class('nn.mil', 'nn.Module')

function mil:__init(disp_max)
   parent.__init(self)
   
   self.disp_max = disp_max;
   
   self.I_maxInRow_refPos = torch.Tensor() 
   self.I_maxInRow_refNeg = torch.Tensor() 

   self.I_maxInCol_refPos = torch.Tensor() 
   self.I_maxInCol_negPos = torch.Tensor() 

end

function mil:updateOutput(input)
  
  local E_refPos, E_refNeg, E_negPos = unpack(input)
  local dim = E_refPos:size(1)
  local outdim = dim - (1 + self.disp_max) + 1; 
  
  local E_maxInRow_refPos, I_maxInRow_refPos = torch.max(E_refPos, 2)
  local E_maxInRow_refNeg, I_maxInRow_refNeg = torch.max(E_refNeg, 2)
  
  E_maxInRow_refPos:view(E_maxInRow_refPos, dim)
  E_maxInRow_refNeg:view(E_maxInRow_refNeg, dim)
  I_maxInRow_refPos:view(I_maxInRow_refPos, dim)
  I_maxInRow_refNeg:view(I_maxInRow_refNeg, dim)
  
  -- cut top disp_max rows
  E_maxInRow_refPos = E_maxInRow_refPos[{{1+self.disp_max, dim}}]:squeeze() 
  E_maxInRow_refNeg = E_maxInRow_refNeg[{{1+self.disp_max, dim}}]:squeeze()
  self.I_maxInRow_refPos = I_maxInRow_refPos[{{1+self.disp_max, dim}}]:squeeze():cuda() 
  self.I_maxInRow_refNeg = I_maxInRow_refNeg[{{1+self.disp_max, dim}}]:squeeze():cuda()
    
  local E_maxInCol_refPos, I_maxInCol_refPos = torch.max(E_refPos, 1) 
  local E_maxInCol_negPos, I_maxInCol_negPos = torch.max(E_negPos, 1) 
    
  E_maxInCol_refPos:view(E_maxInCol_refPos, dim)
  E_maxInCol_negPos:view(E_maxInCol_negPos, dim)
  I_maxInCol_refPos:view(I_maxInCol_refPos, dim)
  I_maxInCol_negPos:view(I_maxInCol_negPos, dim)
  
  -- cut last disp_max cols
  E_maxInCol_refPos = E_maxInCol_refPos[{{1, dim-self.disp_max}}] 
  E_maxInCol_negPos = E_maxInCol_negPos[{{1, dim-self.disp_max}}]
  self.I_maxInCol_refPos = I_maxInCol_refPos[{{1, dim-self.disp_max}}]:cuda()
  self.I_maxInCol_negPos = I_maxInCol_negPos[{{1, dim-self.disp_max}}]:cuda()
    
  
  self.output = {{E_maxInRow_refPos, E_maxInRow_refNeg},  -- mil fwd 
                 {E_maxInCol_refPos, E_maxInCol_negPos}}  -- mil bwd, every tensor is width-disp_max-hpatch*2
                   
  return self.output 
    
end

function mil:updateGradInput(input, gradOutput)
   
  local fwdMil, bwdMil = unpack(gradOutput)
 
  local ograd_maxInRow_refPos, ograd_maxInRow_refNeg = unpack(fwdMil)
  local ograd_maxInCol_refPos, ograd_maxInCol_negPos = unpack(bwdMil)
 
  local E_refPos, E_refNeg, E_negPos = unpack(input)
  local dim = E_refPos:size(1) 
  
  self.igrad_negPos = self.igrad_negPos or E_negPos:clone()
  self.igrad_refPos = self.igrad_refPos or E_refPos:clone()
  self.igrad_refNeg = self.igrad_refNeg or E_refNeg:clone()
  self.igrad_negPos:zero()
  self.igrad_refPos:zero()
  self.igrad_refNeg:zero()
  
  local igrad_refPos_vec = self.igrad_refPos:view(dim*dim) 
  local igrad_refNeg_vec = self.igrad_refNeg:view(dim*dim) 
  local igrad_negPos_vec = self.igrad_negPos:view(dim*dim) 
  local idx;
  
  local row = torch.range(1+self.disp_max, dim):cuda()
  local col = torch.range(1, dim - self.disp_max):cuda()
  
  -- mil fwd
  idx = (self.I_maxInRow_refPos) + (row-1)*dim;
  igrad_refPos_vec:indexAdd(1, idx, ograd_maxInRow_refPos)
  idx = (self.I_maxInRow_refNeg) + (row-1)*dim;
  igrad_refNeg_vec:indexAdd(1, idx, ograd_maxInRow_refNeg)
  
  -- mil bwd
  idx = col + (self.I_maxInCol_refPos-1)*dim;
  igrad_refPos_vec:indexAdd(1, idx, ograd_maxInCol_refPos)
  idx = col + (self.I_maxInCol_negPos-1)*dim;
  igrad_negPos_vec:indexAdd(1, idx, ograd_maxInCol_negPos)
    
  self.gradInput = {self.igrad_refPos, self.igrad_refNeg, self.igrad_negPos}
   
  return self.gradInput
      
end
