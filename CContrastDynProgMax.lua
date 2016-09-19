
--[[
  Given h x w tensor as an _input_, module outputs table with two tensors: _rowDynProg_ and _rowMax_
    
    _rowDynProg_ is h tensor, that consists row wise dynamic programming solutions 
    _rowMax_ is h tensor, that consists of row-wise maximums that are no closer to 
      the first maximum than _distMin_

--]]

local contrastDynProgMax, parent = torch.class('nn.contrastDynProgMax', 'nn.Module')

function contrastDynProgMax:__init(distMin)
   parent.__init(self)
   self.distMin = distMin
   -- these vector store indices of for Dyn Prog solution and row-wise maximums
   self._indicesDynProg = torch.Tensor()
   self._indicesMax = torch.Tensor()
end

function contrastDynProgMax:updateOutput(input)
   
  local _input = input:clone()
  local _outputDynProg = torch.Tensor(input:size(1),1)
   
   -- compute dynamic programming solution 
  local aE, aWay = self:accumulate(_input)
  self._indicesDynProg = nn.utils.addSingletonDimension(self:trace(aE, aWay):long(),2)
  local _outputDynProg =  _input:gather(2,  self._indicesDynProg)
  
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
      
   self.output = torch.cat({_outputDynProg,_outputMax},2)
      
   return self.output
end


function contrastDynProgMax:trace(aEnergy, aWay)
  
  local h = aEnergy:size(1)
  local w = aEnergy:size(2)
  
  local trj = torch.zeros(h)
  local E
  
  -- find best energy in last column
  E, trj[h] = torch.max(aEnergy[{{h},{}}],2)
  
  -- propogate best energy back
  for nrow = h-1, 1, -1  do
    trj[nrow] = aWay[{{nrow+1},{trj[nrow+1]}}]
  end

  return trj
end

function contrastDynProgMax:accumulate(energy)
  
  local h = energy:size(1)
  local w = energy:size(2)
  
  -- initialize top row of accumulated energy to energy
  local aEnergy = energy:clone()  
  local aWay = torch.zeros(h, w)
  
  -- go from top row down, computing best accumulated energy in every position 
  for nrow = 2,h do
    
    local maxIdx = torch.Tensor(w)
    local maxVal = torch.Tensor(w)
    maxIdx[1] = 1;
    maxVal[1] = aEnergy[{{nrow-1},{1}}]
    for ncol = 2,w do
        if( maxVal[ncol-1] < aEnergy[{{nrow-1},{ncol}}]:squeeze() ) then
          maxVal[ncol] = aEnergy[{{nrow-1},{ncol}}]
          maxIdx[ncol] = ncol;
        else
          maxIdx[ncol] = maxIdx[ncol-1];
          maxVal[ncol] = maxVal[ncol-1] 
        end
    end
  
    -- nonstrict monotonicity - we can never go right
    for ncol = 1,w do
      local bestEnergy = maxVal[ncol] 
      local bestInd = maxIdx[ncol]
      aEnergy[{{nrow},{ncol}}] = bestEnergy + energy[{{nrow},{ncol}}]
      aWay[{{nrow},{ncol}}] = bestInd;
    end
    
  end
  
  return aEnergy, aWay
end


function contrastDynProgMax:updateGradInput(input, gradOutput)
   
   -- pass input gradient to dyn prog and max 
   self.gradInput:resizeAs(input):zero()
   self.gradInput:scatter(2, self._indicesDynProg, nn.utils.addSingletonDimension(gradOutput:select(2,1),2))
   self.gradInput:scatter(2, self._indicesMax, nn.utils.addSingletonDimension(gradOutput:select(2,2),2))
   
   return self.gradInput
end
