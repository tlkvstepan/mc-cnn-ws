
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

   self.I_maxInCol  = torch.Tensor()
   self.I_2maxInCol = torch.Tensor()

end

function contrastive:updateOutput(input)
  
  local inputFwd = input;
  local inputBwd = input:clone();
    
  local E_maxInRow, I_maxInRow = torch.max(inputFwd, 2)
  local E_maxInCol, I_maxInCol = torch.max(inputBwd, 1)
 
  local dim = input:size(1)
  
  -- mask maximum and all neighbours of the max
  for d = -self.th_sup, self.th_sup do
    
     -- fwd
     local ind =  I_maxInRow + d
     ind[ind:lt(1)] = 1
     ind[ind:gt(dim)] = dim
     inputFwd = inputFwd:scatter(2, ind, -1/0)
  
     -- bwd 
     ind =  I_maxInCol + d
     ind[ind:lt(1)] = 1
     ind[ind:gt(dim)] = dim
     inputBwd = inputBwd:scatter(1, ind, -1/0)
  
  end
   
  E_maxInRow = E_maxInRow:squeeze()
  I_maxInRow = I_maxInRow:squeeze()
  E_maxInCol = E_maxInCol:squeeze()
  I_maxInCol = I_maxInCol:squeeze() 
   
  local E_2maxInRow, I_2maxInRow = torch.max(inputFwd, 2)
  local E_2maxInCol, I_2maxInCol = torch.max(inputBwd, 1)
  
  E_2maxInRow = E_2maxInRow:squeeze()
  I_2maxInRow = I_2maxInRow:squeeze()
  E_2maxInCol = E_2maxInCol:squeeze()
  I_2maxInCol = I_2maxInCol:squeeze() 
          
  -- cut top disp_max rows (since they might not have true matches)
  E_maxInRow = E_maxInRow[{{1+self.disp_max, dim}}] 
  self.I_maxInRow = I_maxInRow[{{1+self.disp_max, dim}}]:cuda() 
  --
  E_2maxInRow = E_2maxInRow[{{1+self.disp_max, dim}}] 
  self.I_2maxInRow = I_2maxInRow[{{1+self.disp_max, dim}}]:cuda() 
  
  -- cut left colums
  E_maxInCol = E_maxInCol[{{1, dim-self.disp_max}}] 
  self.I_maxInCol = I_maxInCol[{{1, dim-self.disp_max}}]:cuda() 
  --
  E_2maxInCol = E_2maxInCol[{{1, dim-self.disp_max}}]
  self.I_2maxInCol = I_2maxInCol[{{1, dim-self.disp_max}}]:cuda() 
    
  self.output = {{E_maxInRow, E_2maxInRow}, {E_maxInCol, E_2maxInCol}}
      
  return self.output
end

function contrastive:updateGradInput(input, gradOutput)
  
  local contrastiveFwd, contrastiveBwd = unpack(gradOutput)
  
  local ograd_maxInRow, ograd_2maxInRow = unpack(contrastiveFwd)
  local ograd_maxInCol, ograd_2maxInCol = unpack(contrastiveBwd)
   
  -- pass input gradient to max and second max 
  self.gradInput:resizeAs(input):zero()
  local dim = input:size(1)
     
  local gradInput_vec = self.gradInput:view(dim*dim) 
  local idx;
  
  -- fwd
  local row = torch.range(1+self.disp_max, dim):cuda()
   
  idx = self.I_maxInRow + (row-1)*dim;
  gradInput_vec:indexAdd(1, idx, ograd_maxInRow:squeeze())
  --  
  idx = self.I_2maxInRow + (row-1)*dim;
  gradInput_vec:indexAdd(1, idx, ograd_2maxInRow:squeeze())
  
  -- bwd
  local col = torch.range(1, dim - self.disp_max):cuda()
  
  idx = col + (self.I_maxInCol-1)*dim;
  gradInput_vec:indexAdd(1, idx, ograd_maxInCol:squeeze())
  --  
  idx = col + (self.I_2maxInCol-1)*dim;
  gradInput_vec:indexAdd(1, idx, ograd_2maxInCol:squeeze())
    
  return self.gradInput
end

