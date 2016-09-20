
--[[
  Given h x w tensor as an _input_, module outputs _rowDynProg_ 
    
    _rowDynProg_ is h tensor, that consists of row-wise dynamic programming solutions 
    
--]]

local dynProg, parent = torch.class('nn.dynProg', 'nn.Module')

function dynProg:__init(distMin)
   parent.__init(self)
   self.distMin = distMin
   -- these vector store indices of for Dyn Prog solution and row-wise maximums
   self._indicesDynProg = torch.Tensor()
end

function dynProg:updateOutput(input)
  
  local _input = input:clone():double() -- regardless of input our module always computes dyprog on CPU
  local _outputDynProg = torch.Tensor(input:size(1),1)
   
   -- compute dynamic programming solution 
  local aE =  torch.FloatTensor(input:size(1),input:size(2))
  local aP =  torch.FloatTensor(input:size(1),input:size(2))
  self._indicesDynProg = torch.FloatTensor(input:size(1))
  local _outputDynProg =  torch.FloatTensor(input:size(1))

  dynprog.compute(input:float(), aE, aP, self._indicesDynProg, _outputDynProg);
  
  _outputDynProg=_outputDynProg:double()
  self._indicesDynProg = nn.utils.addSingletonDimension(self._indicesDynProg:long(),2)
  self._indicesDynProg = self._indicesDynProg + 1; -- C indexes start from 0
  
  -- if input is cuda tensor than do only dprog on CPU 
  if input:type() == "torch.CudaTensor" then
      self.output = _outputDynProg:cuda()
      self._indicesDynProg = self._indicesDynProg:cuda()
  else
      self.output = _outputDynProg
  end  
  return self.output
end

function dynProg:updateGradInput(input, gradOutput)
   
   -- pass input gradient to dyn prog and max 
   self.gradInput:resizeAs(input):zero()
   self.gradInput:scatter(2, self._indicesDynProg, nn.utils.addSingletonDimension(gradOutput,2))
   
   return self.gradInput
end
