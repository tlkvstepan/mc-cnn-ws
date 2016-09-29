
--[[
    As input module receives table of 3  w x w tensors: _refPos_, _refNeg_, _negPos_ 
    Module outputs table of 2: _fwdMatches_, _bwdMatches_ 
      _fwdMatches_ contains energies of dprog matches for _refPos_ and _refNeg_
      _bwdMatches_ contains energies of dprog matches for _posRef_ and _posNeg_
]]--

local milDprog, parent = torch.class('nn.milDprog', 'nn.Module')

function milDprog:__init()
   parent.__init(self)
   self.fwdRows = torch.Tensor()  -- columns non-occluded simultaneously in refPos and refNeg
 --  self.
end

function milDprog:updateOutput(input)
  
  E_refPos, E_refNeg, E_negPos = unpack(input)
  local dim = E_refPos:size(2)
  local E_refPos = E_refPos:float()
  local E_refNeg = E_refNeg:float()
  local E_negPos = E_negPos:float()
  
  local path_refPos = E_refPos:clone():zero():float()
  local pathNonOcc_refPos = E_refPos:clone():zero():float()
  local path_refNeg = E_refNeg:clone():zero():float()
  local pathNonOcc_refNeg = E_refNeg:clone():zero():float()
  local path_negPos = E_negPos:clone():zero():float()
  local pathNonOcc_negPos = E_negPos:clone():zero():float()
  local pathE = E_negPos:clone():zero():float() 
  local pathLen = E_negPos:clone():zero():float()
    
  dprog.compute(E_refPos, path_refPos, pathE:zero(), pathLen:zero())
  dprog.compute(E_refNeg, path_refNeg, pathE:zero(), pathLen:zero())
  dprog.compute(E_negPos, path_negPos, pathE:zero(), pathLen:zero())
  
  dprog.findNonoccPath(path_refPos, pathNonOcc_refPos)
  dprog.findNonoccPath(path_refNeg, pathNonOcc_refNeg)
  dprog.findNonoccPath(path_negPos, pathNonOcc_negPos)
  
  pathNonOcc_refPos_ = pathNonOcc_refPos:clone()
  
  -- find rows that simultaneously are not occluded in refPos and refNeg solution
  local mask = torch.repeatTensor(pathNonOcc_refPos:max(2), 1, dim)
  mask:cmul(torch.repeatTensor(pathNonOcc_refNeg:max(2), 1, dim))
  pathNonOcc_refPos:cmul(mask)
  pathNonOcc_refNeg:cmul(mask)
  
  -- we can find rowwise match energy using masking
  local dprogE_refPos = E_refPos[pathNonOcc_refPos]:clone()
  local dprogE_refNeg = E_refNeg[pathNonOcc_refNeg]:clone()
  
  -- find match indices 
  local indices_refPos = pathNonOcc_refPos:nonzero() -- valid matches
--  local self.fwdRows = indices_refPos[{{},{1}}]:clone():squeeze():float():add(-1) -- C++ style
 -- local self.fwdCols_refPos = indices_refPos[{{},{2}}]:clone():squeeze():float():add(-1)
--  local indices_refNeg = pathNonOcc_refNeg:nonzero() -- valid matches
 -- local self.fwdCols_refNeg = indices_refPos[{{},{2}}]:clone():squeeze():float():add(-1)
 
  
  
  
--  local indices = pathNonOcc_refPos:nonzero() -- valid matches
--  local dprogErefPos = E_refPos[pathNonOcc_refPos]
--  local rows_refPos = indices[{{},{1}}]:clone():squeeze():float():add(-1) -- C++ style
--  local cols_refPos = indices[{{},{2}}]:clone():squeeze():float():add(-1)
  
--  local indices = pathNonOcc_refNeg:nonzero() -- valid matches
--  local dprogErefNeg = E_refNeg[pathNonOcc_refNeg]
--  local rows_refNeg = indices[{{},{1}}]:clone():squeeze():float():add(-1) -- C++ style
--  local cols_refNeg = indices[{{},{2}}]:clone():squeeze():float():add(-1)
  
  
  
  
  
  
--  local indices = pathNonOcc_refPos:nonzero() -- valid matches
--  local dprogErefPos = E_refPos[pathNonOcc_refPos]
--  local rows_refPos = indices[{{},{1}}]:clone():squeeze():float():add(-1) -- C++ style
--  local cols_refPos = indices[{{},{2}}]:clone():squeeze():float():add(-1)
  
  

--  dprogE = E[pathNonOcc:byte()]:clone()
--  dim = dprogE:numel()
--  self.rowwiseMaxI = torch.zeros(dim):float()
--  rowwiseMaxE = torch.zeros(dim):float()
--  self.colwiseMaxI = torch.zeros(dim):float()
--  colwiseMaxE = torch.zeros(dim):float()
--  dprog.findMaxForRows(E_masked, self.rows, self.rowwiseMaxI, rowwiseMaxE)
--  dprog.findMaxForCols(E_masked, self.cols, self.colwiseMaxI, colwiseMaxE)

 
--  dprogE = dprogE:double()
--  rowwiseMaxE = rowwiseMaxE:double()
--  colwiseMaxE = colwiseMaxE:double()
  
--  -- if cuda is on than transfer all to cuda 
--  if input:type() == "torch.CudaTensor" then
 
--    dprogE = dprogE:cuda()
--    rowwiseMaxE = rowwiseMaxE:cuda()
--    colwiseMaxE = colwiseMaxE:cuda()
    
--  end

--  self.output = torch.cat({nn.utils.addSingletonDimension(dprogE,2), nn.utils.addSingletonDimension(rowwiseMaxE,2), nn.utils.addSingletonDimension(colwiseMaxE,2)}, 2)
   
  return self.output
end

function milDprog:updateGradInput(input, gradOutput)
   
   -- pass input gradient to dyn prog and max 
   self.gradInput = self.gradInput:resizeAs(input):zero():float()
   
   dprog.collect(self.gradInput, gradOutput:select(2,1):float(), self.cols, self.rows)
   dprog.collect(self.gradInput, gradOutput:select(2,2):float(), self.rowwiseMaxI, self.rows)
   dprog.collect(self.gradInput, gradOutput:select(2,3):float(), self.cols, self.colwiseMaxI)
    
   self.gradInput = self.gradInput:double() 
   if input:type() == "torch.CudaTensor" then 
    self.gradInput = self.gradInput:cuda()
   end
      
   return self.gradInput
end
