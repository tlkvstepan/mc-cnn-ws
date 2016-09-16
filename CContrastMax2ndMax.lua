
--[[
  Given h x w tensor as an _input_, module outputs table with two tensors: _rowMax_ and _row2ndMax_
    
    _rowMax_ is h tensor, that consists of row-wise maximums 
    _row2ndMax_ is h tensor, that consists of row-wise 2nd order maximums that are no closer to 
      the first maximum than _distMin_

--]]

local contrastMax2ndMax, parent = torch.class('nn.contrastMax2ndMax', 'nn.Module')

function contrastMax2ndMax:__init(distMin)
   parent.__init(self)
   self.distMin = distMin
   -- these vector store indices of 1st and 2nd row-wise maximums
   self._indicesMax = torch.Tensor()
   self._indices2ndMax = torch.Tensor()
end

function contrastMax2ndMax:updateOutput(input)
   
   local _input = input:clone()
   local _outputMax
   
   -- compute maximum
   _outputMax, self._indicesMax = torch.max(_input, 2)
   
   -- mask maximum and all neighbours of the max
   for dist = -self.distMin, self.distMin do
     local ind =  self._indicesMax + dist
     ind[ind:lt(1)] = 1
     ind[ind:gt(_input:size(2))] = _input:size(2)
     _input = _input:scatter(2, ind, -1/0)
   end
   
   -- compute second maximum
   _output2ndMax, self._indices2ndMax = torch.max(_input, 2)
      
   self.output = torch.cat({_outputMax,_output2ndMax},2)
      
   return self.output
end

function contrastMax2ndMax:updateGradInput(input, gradOutput)
   
   -- pass input gradient to max and second max 
   self.gradInput:resizeAs(input):zero()
   self.gradInput:scatter(2, self._indicesMax, nn.utils.addSingletonDimension(gradOutput:select(2,1),2))
   self.gradInput:scatter(2, self._indices2ndMax, nn.utils.addSingletonDimension(gradOutput:select(2,2),2))
   
   return self.gradInput
end

