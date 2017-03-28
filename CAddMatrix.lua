local addMatrix, Parent = torch.class('nn.addMatrix', 'nn.Module')

function addMatrix:__init(m)
   Parent.__init(self)
   self.m = m
end

function addMatrix:updateOutput(input)
    self.output = input + self.m
    return self.output
end

function addMatrix:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput;
  return gradOutput;
end
