local copyElements, parent = torch.class('nn.copyElements', 'nn.Module')

function copyElements:__init(iSize, oSize, iIndex, oIndex)
  
  parent.__init(self)
  
  self.iIndex = iIndex;
  self.oIndex = oIndex;
  self.iSize = iSize;
  self.oSize = oSize;        
  
end

function copyElements:setup(iSize, oSize, iIndex, oIndex)
  
  self.iIndex = iIndex;
  self.oIndex = oIndex;
  self.iSize = iSize;
  self.oSize = oSize;

  
end

function copyElements:updateOutput(input)
  
  
  self.output:resize(self.oSize):zero()
    
  local input_vec = input:view(torch.prod(torch.LongTensor(self.iSize)))
  local output_vec = self.output:view(torch.prod(torch.LongTensor(self.oSize)))
    
  output_vec:indexCopy(1, self.oIndex, input_vec:index(1, self.iIndex))
  
  return self.output
end

function copyElements:updateGradInput(input, gradOutput)
  
  self.gradInput:resize(self.iSize):zero();
  
  local gradInput_vec = self.gradInput:view(torch.prod(torch.LongTensor(self.iSize)))
  local gradOutput_vec = gradOutput:view(torch.prod(torch.LongTensor(self.oSize)))
  
  gradInput_vec:indexCopy(1, self.iIndex, gradOutput_vec:index(1, self.oIndex))
     
  return self.gradInput

end
