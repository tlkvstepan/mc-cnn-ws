local MaxM, parent = torch.class('nn.MaxM', 'nn.Module')

function MaxM:__init(dimension, n, r, nInputDims)
   parent.__init(self)
   self.n = n
   if r == nil then r = 0 end
   self.r = r
   dimension = dimension or 1
   self.dimension = dimension
   -- do not assign default value to nInputDims or it will break backward compatibility
   self.nInputDims = nInputDims
end

function MaxM:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input:dim() + dimension + 1
   elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end

function MaxM:_lazyInit()
   self._output = self._output or self.output.new()
   self._indices = self._indices or
      (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaTensor() or torch.LongTensor())
end

function MaxM:updateOutput(input)
   self:_lazyInit()
   local dimension = self:_getPositiveDimension(input)
  
   local _input  = input:clone()
   for i = 1, self.n do
      torch.max(self._output, self._indices, _input, dimension)
      _input = _input:scatter(dimension, self._indices, -1/0)
      for j = -self.r,self.r do
          local ind =  self._indices + j;
          ind[ind:lt(1)] = 1;
          ind[ind:gt(input:size(2))] = input:size(2);
          _input = _input:scatter(dimension, ind, -1/0)
      end
   end
      
   if input:dim() > 1 then
     self.output:set(self._output:select(dimension, 1))
   else
     self.output:set(self._output)
   end
   return self.output
end

function MaxM:updateGradInput(input, gradOutput)
   self:_lazyInit()
   local dimension = self:_getPositiveDimension(input)
   local gradOutputView
   if input:dim() > 1 then
     gradOutputView = nn.utils.addSingletonDimension(gradOutput, dimension)
   else
     gradOutputView = gradOutput
   end
   self.gradInput:resizeAs(input):zero():scatter(dimension, self._indices, gradOutputView)
   return self.gradInput
end

function MaxM:type(type, tensorCache)
  -- torch.MaxM expects a LongTensor as indices, whereas cutorch.MaxM expects a CudaTensor.
  if type == 'torch.CudaTensor' then
    parent.type(self, type, tensorCache)
  else
    -- self._indices must be a LongTensor. Setting it to nil temporarily avoids
    -- unnecessary memory allocations.
    local indices
    indices, self._indices = self._indices, nil
    parent.type(self, type, tensorCache)
    self._indices = indices and indices:long() or nil
  end
  return self
end

function MaxM:clearState()
   nn.utils.clear(self, '_indices', '_output')
   return parent.clearState(self)
end
