local stereoCost, parent = torch.class('nn.stereoCost', 'nn.Module')

function stereoCost:__init(width, dispMax)
   -- simNet  - network that takes two desciptors and returns similarity measure
      
   parent.__init(self)
   
   self.dispMax = dispMax;
   self.width = width;
   
   -- these vector store indices of for Dyn Prog solution and row-wise maximums
   --self.cols = torch.Tensor()
   --self.rows = torch.Tensor()
   
end

function stereoCost:updateOutput(input)
  -- input is nb_features x nb_patches
  
  local dim = input:size(1)
  local nb_patches = input:size(2)
  
  
  
  
  
  return self.output
end

function stereoCost:updateGradInput(input, gradOutput)
      
   local fwd, bwd  = unpack(gradOutput)
   local gradOutput_dprog1, gradOutput_row = unpack(fwd)
   local gradOutput_dprog2, gradOutput_col = unpack(bwd)
   
   -- pass input gradient to dyn prog and max 
   self.gradInput = self.gradInput:resizeAs(input):zero() -- same type as input
   local dim = input:size(1);
   local gradInput_vec = self.gradInput:view(dim*dim) 
   local idx;

   idx = (self.cols) + (self.rows-1)*dim;
   gradInput_vec:indexAdd(1, idx, gradOutput_dprog1)
   gradInput_vec:indexAdd(1, idx, gradOutput_dprog2)
  
   idx = (self.rowwiseMaxI) + (self.rows-1)*dim;
   gradInput_vec:indexAdd(1, idx, gradOutput_row)
   
   idx = (self.cols) + (self.colwiseMaxI-1)*dim;
   gradInput_vec:indexAdd(1, idx, gradOutput_col)
   
   return self.gradInput
end
