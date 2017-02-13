
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
   self.update = true;
   self.arr_idx = arr_idx:cuda():cudaLong()

end

function headNetMulti:updateOutput(input)
  
  local arr1, arr2 = unpack(input)
  local dim   = arr1:size(1);
    
  local nb_pairs = self.arr_idx:size(1)
  local row = self.arr_idx:select(2,1)
  local col = self.arr_idx:select(2,2)
  
  self.output = torch.CudaTensor(dim, dim):zero() 
    
  local in1 = arr1:index(1, row)  
  local in2 = arr2:index(1, col)  
 
  self.headNet:forward{ in1, in2 }  
    
  local output_vec = self.output:view(dim*dim) 
  local idx = col + (row-1)*dim;
  output_vec:indexAdd(1, idx:long(), self.headNet.output)
  
  -- if we store states for all pairs we will run out of memory
  self.headNet:clearState()   
  
  return self.output 
    
end



function headNetMulti:updateGradInput(input, gradOutput)
   
  local arr1, arr2 = unpack(input)
  local dim = gradOutput:size(1)
    
  -- after we receive output gradients (which are sparse!) we know what pairs are useful
  local useful_arr_idx = gradOutput:nonzero():cuda()
  if( useful_arr_idx:numel() > 0 ) then    
  do 
    local flat_idx = useful_arr_idx:select(2,2) + (useful_arr_idx:select(2,1) - 1)*dim
    local gradOutput_vec = gradOutput:view(dim*dim)
    self.useful_gradOutput = gradOutput_vec:index(1, flat_idx:cudaLong())
  end

  -- selector nets select usefull elements for cost computation
  local selectorNet1 = nn.Index(1):cuda()
  local selectorNet2 = nn.Index(1):cuda()
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
  
  self.gradInput = {selectorNet1.gradInput[1], selectorNet2.gradInput[1]} 
  self.update = true;
  else
    self.update = false;
    self.gradInput = {arr1:clone():zero(), arr2:clone():zero()}
  end

  return self.gradInput
      
end

function headNetMulti:accGradParameters(input, gradOutput)
  if self.update then
    self.headNet:accGradParameters(self.selectorOutput, self.useful_gradOutput)
  end
end

function headNetMulti:parameters()

  weights, gradWeights = self.headNet:parameters()
  
  return weights, gradWeights 
end