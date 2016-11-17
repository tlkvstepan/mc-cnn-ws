local StereoJoin, parent = torch.class('nn.StereoJoin', 'nn.Module')

function StereoJoin:__init(disp_max)
   parent.__init(self)
   self.disp_max = disp_max
   self.output = torch.CudaTensor() 
end

function StereoJoin:updateOutput(input)
   assert(input:size(1) == 2)
   local disc0 = input[{{1}}]
   local disc1 = input[{{2}}]
   self.output:resize(2, self.disp_max, input:size(3), input:size(4))
   adcensus.StereoJoin(disc0, disc1, self.output[{{1}}], self.output[{{2}}])
    
   adcensus.StereoJoin_forward2(input_L, input_R, self.output_L)
   return self.output
end

function StereoJoin:updateGradInput(input, gradOutput)
   
   gradOutput
   
   
   
   return self.output_L
end