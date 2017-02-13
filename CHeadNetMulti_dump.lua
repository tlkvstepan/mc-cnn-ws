
--[[
    
    Input:
    
    
    This module outputs two matrices nb_matches x 2 tensor.
    In first column of this tensor are energies of dprog matches for _refPos_
    In second column of this tensor are energies of dprog matches for _refNeg_
     Note that we contrast refPos with refNeg as well as refPos with negPos
--]]

local headNetMulti, parent = torch.class('nn.headNetMulti', 'nn.Module')

function headNetMulti:__init(arr_idx, headNet )
   
   parent.__init(self)
   
   -- NB! head net should be on GPU for optimal speed
   self.headNet = headNet:clone('weight','bias', 'gradWeight','gradBias')
   
   self.arr_idx = arr_idx

end

function headNetMulti:updateOutput(input)
  
  local arr1, arr2 = unpack(input)
  
  local nb_pairs = self.arr_idx:size(1)
  
  local sizes = torch.LongStorage{arr1:size(1), arr2:size(1)}
  
  self.output = torch.zeros(sizes) 
      
  self.headNet:forward{ arr1:index(1, self.arr_idx:select(2,1)), arr2:index(1, self.arr_idx:select(2,2)) }  
  self.output = self.headNet.output  

  return self.output 
    
end



function headNetMulti:updateGradInput(input, gradOutput)
   
  local arr1, arr2 = unpack(input)
  local dim = gradOutput:size(1)
    
  -- after we receive output gradients (which are sparse!) we know what pairs are useful
  local useful_arr_idx
    
  do 
    useful_arr_idx = gradOutput:nonzero()
    local flat_idx = useful_arr_idx:select(2,2) + (useful_arr_idx:select(2,1) - 1)*dim
    local gradOutput_vec = gradOutput:view(dim*dim)
    self.useful_gradOutput = gradOutput_vec:index(1, flat_idx:long())
  end

  -- selector nets select usefull elements for cost computation
  local selectorNet1 = nn.Index(1)
  local selectorNet2 = nn.Index(1)
  selectorNet1:forward{arr1, useful_arr_idx:select(2,1)}
  selectorNet2:forward{arr2, useful_arr_idx:select(2,2)}
  self.selectorOutput = {selectorNet1.output, selectorNet2.output}
  
  -- we channel selected elements to head net
  self.headNet:forward(self.selectorOutput)
  
  -- compute input gradient of the head net
  self.headNet:updateGradInput(self.selectorOutput, self.useful_gradOutput)
  
  -- compute input gradient of the selector nets
  selectorNet1:backward({arr1, useful_arr_idx:select(2,1)}, self.headNet.gradInput[1])
  selectorNet2:backward({arr2, useful_arr_idx:select(2,2)}, self.headNet.gradInput[2])
  
  self.gradInput = {selectorNet1.gradInput, selectorNet2.gradInput} 
  
  return self.gradInput
      
end

function headNetMulti:accGradParameters(input, gradOutput)
  
  self.headNet:accGradParameters(self.selectorOutput, self.useful_gradOutput)
  
end

function headNetMulti:parameters()

  weights, gradWeights = self.headNet:parameters()
  
  return weights, gradWeights 
end