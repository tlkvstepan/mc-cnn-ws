local stereoPipeline, parent = torch.class('nn.stereoPipeline', 'nn.Module')



function stereoPipeline:__init(max_sup,)
   
--  prm['hpatch'] = 4
--  prm['dispMax'] = 250
  
--  prm['pi1'] = 4
--  prm['pi2'] = 55.72
--  prm['sgm_q1'] = 3
--  prm['sgm_q2'] = 2.5
--  prm['alpha1'] = 1.5
--  prm['tau_so'] = 0.02
  
--  prm['blur_sigma'] =7.74
--  prm['blur_t'] = 5
--  prm['med'] = 3
  
  
  
  parent.__init(self)
  
  
end

function stereoPipeline:updateOutput(input)
  
  vol, trueDisp, actualDisp = unpack(input)
  
  -- vol - dispMax x height x width cost volume
  -- trueDisp - height x width 
  -- actualDisp - height x width 
  
  local dispMax = vol:size(2)
  local height = vol:size(3)
  local width = vol:size(4) 
  
  -- round up disparities
  trueDisp = torch.clamp(trueDisp:add(1):round(), 1, dispMax-1)
  actualDisp = torch.clamp(actualDisp:add(1):round(), 1, dispMax-1)
  
  -- compute costs
  local trueCost = vol:squeeze():gather(1, trueDisp:long())
  local actualCost = vol:squeeze():gather(1, actualDisp:long())
  
  return {trueCost:view(height*widht), actualCost:view(height*widht)}
end

function stereoPipeline:updateGradInput(input, gradOutput)
      
   local trueGradOutput, actualGradOutput  = unpack(gradOutput)
   
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