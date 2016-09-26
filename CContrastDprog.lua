
--[[
  Given h x w tensor as an _input_, module outputs 4 x h tensor
    
    first column  - fDprog
    second column - fMax
    third column  - bDprog
    forth column  - bMax

--]]

local contrastDprog, parent = torch.class('nn.contrastDprog', 'nn.Module')

function contrastDprog:__init(distMin)
   parent.__init(self)
   self.distMin = distMin
   -- these vector store indices of for Dyn Prog solution and row-wise maximums
   self.fDprogIndices = torch.Tensor()
   self.fMaxIndices   = torch.Tensor()
   self.bDprogIndices = torch.Tensor()
   self.bMaxIndices   = torch.Tensor()
end

function contrastDprog:updateOutput(input)
  
  local _input = input:clone():double()
  local _outputDynProg = torch.Tensor(input:size(1),1)
   
  -- compute dynamic programming solution 
  local path = torch.clone(input):zero():float()
  local fDprogCost = torch.zeros(input:size(1)):float()
  local bDprogCost = torch.zeros(input:size(1)):float()
  self.fDprogIndices = torch.zeros(input:size(1)):float()
  self.bDprogIndices = torch.zeros(input:size(1)):float()
  
  dprog.compute(input, path, fDprogCost, bDprogCost, self.fDprogIndices, self.bDprogIndices);




  _outputDynProg=_outputDynProg:double()

  self._indicesDynProg = nn.utils.addSingletonDimension(self._indicesDynProg:long(),2)
  self._indicesDynProg = self._indicesDynProg + 1;
  
  -- mask dyn prog solution and all neighbours of the sol
   for dist = -self.distMin, self.distMin do
     local ind =  self._indicesDynProg + dist
     ind[ind:lt(1)] = 1
     ind[ind:gt(_input:size(2))] = _input:size(2)
     _input = _input:scatter(2, ind, -1/0)
   end
   
   -- compute greedy solution
   local _outputMax
   _outputMax, self._indicesMax = torch.max(_input, 2)
      
   -- if input is cuda tensor than do only dprog on CPU 
   if input:type() == "torch.CudaTensor" then
      self.output = torch.cat({_outputDynProg:cuda(),_outputMax:cuda()},2)
      self._indicesMax = self._indicesMax:cuda()
      self._indicesDynProg = self._indicesDynProg:cuda()
   else
      self.output = torch.cat({_outputDynProg,_outputMax},2)
   end  
   return self.output
end

function contrastDprog:updateGradInput(input, gradOutput)
   
   -- pass input gradient to dyn prog and max 
   self.gradInput:resizeAs(input):zero()
   self.gradInput:scatter(2, self._indicesDynProg, nn.utils.addSingletonDimension(gradOutput:select(2,1),2))
   self.gradInput:scatter(2, self._indicesMax, nn.utils.addSingletonDimension(gradOutput:select(2,2),2))
   
   return self.gradInput
end
