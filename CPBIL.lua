local pbil = torch.class('pbil')

-- prmTab - is table of prm

function pbil:__init(prmValTab, initNumSample, popSize)
  
  self.popSize = popSize or 10
  initNumSample = initNumSample or 100
  
  self.prmProTab = {};
  self.prmValTab = prmValTab;
  
  self.bestPrmCost = 1/0;
  self.bestPrmVal = {}
  
  for prmName, prmVal in pairs(prmValTab) do
    -- for every array of values make array of probabilities
    local N = torch.ones(prmVal:numel())*initNumSample / prmVal:numel()
    self.prmProTab[prmName] = N
  end
  
end

function pbil:sampleHist(hist)
  
  local cumSum = torch.cumsum(hist)
  local sample = math.random()
  for index = 1,cumSum:numel() do
    if( sample < cumSum[index] )  then
      return index
    end
  end
  
end

function pbil:sample()
  
  self.popPrmVal = {}
  self.popPrmIdx = {}
  
  for i = 1, self.popSize do
    local prmVal = {}
    local prmIdx = {}
    for prmName, prmPro in pairs(self.prmProTab) do
      -- for every array of values make array of probabilities
      local index = self:sampleHist(prmPro:mul(1/prmPro:sum()))
      prmVal[prmName]= self.prmValTab[prmName][index]
      prmIdx[prmName]= index;
    end
    self.popPrmIdx[i] = prmIdx
    self.popPrmVal[i] = prmVal
  end
  
  return self.popPrmVal
end


function pbil:update(cost)
  
  local minCost, minIdx = cost:min(1)
  minCost = minCost:squeeze()
  minIdx = minIdx:squeeze()
  
  if self.bestPrmCost > minCost then
    
    self.bestPrmCost = minCost;
    self.bestPrmVal  = self.popPrmVal[minIdx]
  
  end
  
  for prmName, prmProb in pairs(self.prmProTab) do
    
    self.prmProTab[prmName][self.popPrmIdx[minIdx][prmName]] = self.prmProTab[prmName][self.popPrmIdx[minIdx][prmName]] + 1;   
    
  end
  
end