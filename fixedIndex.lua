local fixedIndex, parent = torch.class('nn.fixedIndex', 'nn.Module')

function fixedIndex:__init(dimension, index)
    parent.__init(self)
    self.index = index;
    self.dimension = dimension
--    self.gradInput = {self.gradInput, self.gradInput.new()}
end

function fixedIndex:updateOutput(input)
    self.output:index(input, self.dimension, self.index)
    return self.output
end

function fixedIndex:updateGradInput(input, gradOutput)
    
    --self.gradInput:resize(self.index:size()):zero()
    --local gradInput = self.gradInput[1] -- no gradient for the index variable
    self.gradInput:resizeAs(input):zero()
    self.gradInput:indexAdd(self.dimension, self.index, gradOutput)
    
    return self.gradInput
end

function fixedIndex:clearState()
    self.gradInput[1]:set()
    self.gradInput[2]:set()
    self.output:set()
    return self
end