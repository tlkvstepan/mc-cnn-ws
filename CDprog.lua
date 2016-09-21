
--[[
  Given h x w tensor as an _input_, module outputs _rowDprog_ 
    
    _rowDprog_ is h tensor, that consists of row-wise dynamic programming solutions 
    
--]]

local Dprog, parent = torch.class('nn.Dprog', 'nn.Module')

function Dprog:__init(distMin)
   parent.__init(self)
   self.distMin = distMin
   -- these vector store indices of for Dyn Prog solution and row-wise maximums
   self._indicesDprog = torch.Tensor()
end

function Dprog:updateOutput(input)
  
  local _input = input:clone():double() -- regardless of input our module always computes dyprog on CPU
  local _outputDprog = torch.Tensor(input:size(1),1)
   
   -- compute dynamic programming solution 
  local aE =  torch.FloatTensor(input:size(1),input:size(2))
  local aP =  torch.FloatTensor(input:size(1),input:size(2))
  self._indicesDprog = torch.FloatTensor(input:size(1))
  local _outputDprog =  torch.FloatTensor(input:size(1))

  dprog.compute(input:float(), aE, aP, self._indicesDprog, _outputDprog);
  
  _outputDprog=_outputDprog:double()
  self._indicesDprog = nn.utils.addSingletonDimension(self._indicesDprog:long(),2)
  self._indicesDprog = self._indicesDprog + 1; -- C indexes start from 0
  
  -- if input is cuda tensor than do only Dprog on CPU 
  if input:type() == "torch.CudaTensor" then
      self.output = _outputDprog:cuda()
      self._indicesDprog = self._indicesDprog:cuda()
  else
      self.output = _outputDprog
  end  
  return self.output
end

function Dprog:updateGradInput(input, gradOutput)
   
   -- pass input gradient to dyn prog and max 
   self.gradInput:resizeAs(input):zero()
   self.gradInput:scatter(2, self._indicesDprog, nn.utils.addSingletonDimension(gradOutput,2))
   
   return self.gradInput
end
