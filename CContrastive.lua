
--[[
  Given h x w tensor as an _input_, module outputs table with two tensors: _rowMax_ and _row2ndMax_
    
    _rowMax_ is h tensor, that consists of row-wise maximums 
    _row2ndMax_ is h tensor, that consists of row-wise 2nd order maximums that are no closer to 
      the first maximum than _distMin_

--]]

local contrastive, parent = torch.class('nn.contrastive', 'nn.Module')

function contrastive:__init(th_sup, disp_max)
   parent.__init(self)
   
   self.th_sup = th_sup
   self.disp_max = disp_max
   
   self.I_maxInRow  = torch.Tensor()
   self.I_2maxInRow = torch.Tensor()

end

function contrastive:updateOutput(input)
   
  local E_maxInRow, I_maxInRow = torch.max(input, 2)
  local dim = input:size(1)
  
  -- mask maximum and all neighbours of the max
  for d = -self.th_sup, self.th_sup do
    
     local ind =  I_maxInRow + d
     ind[ind:lt(1)] = 1
     ind[ind:gt(dim)] = dim
     input = input:scatter(2, ind, -1/0)
  
  end
   
  local E_2maxInRow, I_2maxInRow = torch.max(input, 2)
  
  -- cut top disp_max rows (since they might not have true matches)
  E_maxInRow = E_maxInRow[{{1+self.disp_max, dim}}]:squeeze() 
  self.I_maxInRow = I_maxInRow[{{1+self.disp_max, dim}}]:squeeze():cuda() 
  --
  E_2maxInRow = E_2maxInRow[{{1+self.disp_max, dim}}]:squeeze() 
  self.I_2maxInRow = I_2maxInRow[{{1+self.disp_max, dim}}]:squeeze():cuda() 
  
  self.output = {E_maxInRow, E_2maxInRow}
      
  return self.output
end

function contrastive:updateGradInput(input, gradOutput)
   
  local ograd_maxInRow, ograd_2maxInRow = unpack(gradOutput)
   
      -- pass input gradient to max and second max 
  self.gradInput:resizeAs(input):zero()
  local dim = input:size(1)
     
  local gradInput_vec = self.gradInput:view(dim*dim) 
  local idx;
  
  local row = torch.range(1+self.disp_max, dim):cuda()
  
  idx = self.I_maxInRow + (row-1)*dim;
  gradInput_vec:indexAdd(1, idx, ograd_maxInRow:squeeze())
  
  idx = self.I_2maxInRow + (row-1)*dim;
  gradInput_vec:indexAdd(1, idx, ograd_2maxInRow:squeeze())
   
  return self.gradInput
end

