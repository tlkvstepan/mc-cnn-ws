
--[[
    As input module receives 3 distance matrices: _refPos_ _refNeg_ and _posNeg_
    This module outputs two matrices nb_matches x 2 tensor.
    In first column of this tensor are energies of dprog matches for _refPos_
    In second column of this tensor are energies of dprog matches for _refNeg_
    
    Note that we contrast refPos with refNeg as well as refPos with negPos
--]]

local milDprog, parent = torch.class('nn.milDprog', 'nn.Module')

function milDprog:__init()
   parent.__init(self)
   self.activeRowsRef = torch.Tensor() 
   self.matchColRefPos= torch.Tensor() 
   self.matchColRefNeg= torch.Tensor()
   ----
   self.activeRowsPos = torch.Tensor()
   self.matchColPosRef =torch.Tensor()
   
end

function milDprog:updateOutput(input)
  
  E_refPos, E_refNeg, E_negPos= unpack(input)
  
  local E_refPos = E_refPos:clone():float()
  local E_refNeg = E_refNeg:clone():float()
  local E_negPos= E_negPos:clone():float()
  local dim = E_negPos:size(2)
  
  local path_refPos = E_refPos:clone():zero():float()
  local pathNonOcc_refPos = E_refPos:clone():zero():float()
  local path_refNeg = E_refNeg:clone():zero():float()
  local pathNonOcc_refNeg = E_refNeg:clone():zero():float()
  local path_negPos= E_negPos:clone():zero():float()
  local pathNonOcc_negPos= E_negPos:clone():zero():float()
  local pathE = E_negPos:clone():zero():float() 
  local pathLen = E_negPos:clone():zero():float()
    
  dprog.compute(E_refPos, path_refPos, pathE:zero(), pathLen:zero())
  dprog.compute(E_refNeg, path_refNeg, pathE:zero(), pathLen:zero())
  dprog.compute(E_negPos, path_negPos, pathE:zero(), pathLen:zero())
  
  dprog.findNonoccPath(path_refPos, pathNonOcc_refPos)
  dprog.findNonoccPath(path_refNeg, pathNonOcc_refNeg)
  dprog.findNonoccPath(path_negPos, pathNonOcc_negPos)
  
  pathNonOcc_refPos = pathNonOcc_refPos:byte() 
  pathNonOcc_refNeg = pathNonOcc_refNeg:byte() 
  pathNonOcc_negPos = pathNonOcc_negPos:byte() 
  
  
  pathNonOcc_posNeg = pathNonOcc_negPos:clone():t()   
  pathNonOcc_posRef = pathNonOcc_refPos:clone():t()
  
  -- find rows that simultaneously are not occluded in refPos and refNeg solution
  local maskFwd = torch.repeatTensor(pathNonOcc_refPos:max(2), 1, dim)
  maskFwd:cmul(torch.repeatTensor(pathNonOcc_refNeg:max(2), 1, dim))
  pathNonOcc_refPos:cmul(maskFwd)
  pathNonOcc_refNeg:cmul(maskFwd)
    
  local fwd  
  if not torch.all(maskFwd:eq(0))  then
    local indices = pathNonOcc_refPos:nonzero() -- valid matches
    local dprogRefPosE = E_refPos[pathNonOcc_refPos]:clone()
    self.activeRowsRef = indices[{{},{1}}]:clone():squeeze():float():add(-1) -- C++ style
    self.matchColRefPos = indices[{{},{2}}]:clone():squeeze():float():add(-1)
    local indices = pathNonOcc_refNeg:nonzero() -- valid matches
    local dprogRefNegE = E_refNeg[pathNonOcc_refNeg]:clone()
    self.matchColRefNeg = indices[{{},{2}}]:clone():squeeze():float():add(-1)
  
    dprogRefNegE = nn.utils.addSingletonDimension(dprogRefNegE:double(),2)
    dprogRefPosE = nn.utils.addSingletonDimension(dprogRefPosE:double(),2)
    
        -- if cuda is on than transfer all to cuda 
    if input[1]:type() == "torch.CudaTensor" then
      
      dprogRefNegE = dprogRefNegE:cuda()
      dprogRefPosE = dprogRefPosE:cuda()
      
    end
     
    fwd = torch.cat({dprogRefPosE, dprogRefNegE}, 2)
  
  end
  
  -- find rows that are simultaneously not occluded in posRef and posNeg
  local maskBwd = torch.repeatTensor(pathNonOcc_posRef:max(2), 1, dim)
  maskBwd:cmul(torch.repeatTensor(pathNonOcc_posNeg:max(2), 1, dim))
  pathNonOcc_posRef:cmul(maskBwd)
  pathNonOcc_posNeg:cmul(maskBwd)
  
  local bwd
  if not torch.all(maskBwd:eq(0))   then 
    local indices = pathNonOcc_posRef:nonzero() -- valid matches
    local dprogPosRefE = E_refPos:t()[pathNonOcc_posRef]:clone()
    self.activeRowsPos = indices[{{},{1}}]:clone():squeeze():float():add(-1) -- C++ style
    self.matchColPosRef = indices[{{},{2}}]:clone():squeeze():float():add(-1)
    indices = pathNonOcc_posNeg:nonzero() -- valid matches
    local dprogPosNegE = E_negPos:t()[pathNonOcc_posNeg]:clone()
    self.matchColPosNeg = indices[{{},{2}}]:clone():squeeze():float():add(-1)
    
    -- output is double
    dprogPosNegE = nn.utils.addSingletonDimension(dprogPosNegE:double(),2)
    dprogPosRefE = nn.utils.addSingletonDimension(dprogPosRefE:double(),2)
  
    -- if cuda is on than transfer all to cuda 
    if input[1]:type() == "torch.CudaTensor" then
      
      dprogPosNegE = dprogPosNegE:cuda()
      dprogPosRefE = dprogPosRefE:cuda()
      
    end
    
    -- make output tensor
    bwd = torch.cat({dprogPosRefE, dprogPosNegE}, 2)
      
  end
    
  self.output = {fwd, bwd}
    
  -- note:
  -- 1. sometimes fwd or bwd can be empty
  -- 2. number of elements in fwd and bwd can be differenet 
  
  return self.output
    
end

function milDprog:updateGradInput(input, gradOutput)
   
  local gradOutput_fwd, gradOutput_bwd = unpack(gradOutput)
  local E_refPos, E_refNeg, E_negPos = unpack(input)
  
  
  local dprogPosRefE_grad = gradOutput_bwd[{{},{1}}]:squeeze():float()
  local dprogPosNegE_grad = gradOutput_bwd[{{},{2}}]:squeeze():float()
  
  local dprogRefPosE_grad = gradOutput_fwd[{{},{1}}]:squeeze():float()
  local dprogRefNegE_grad = gradOutput_fwd[{{},{2}}]:squeeze():float()  
  
   -- pass input gradient to dyn prog and max 
   self.gradInput[1] = self.gradInput:resizeAs(E_refPos):zero():float()
   self.gradInput[2] = self.gradInput:resizeAs(E_refNeg):zero():float()
   self.gradInput[3] = self.gradInput:resizeAs(E_negPos):zero():float()
   
   dprog.collect(self.gradInput_refPos, gradOutput:select(2,1):float(), self.cols, self.rows)
   dprog.collect(self.gradInput_refNeg, gradOutput:select(2,2):float(), self.rowwiseMaxI, self.rows)
   dprog.collect(self.gradInput_negPos, gradOutput:select(2,3):float(), self.cols, self.colwiseMaxI)
    
   self.gradInput = self.gradInput:double() 
   
   if input:type() == "torch.CudaTensor" then 
    self.gradInput = self.gradInput:cuda()
   end
      
   return self.gradInput
end
