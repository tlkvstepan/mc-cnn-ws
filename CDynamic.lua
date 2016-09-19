require 'gnuplot'

function trace(aEnergy, aWay)
  
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

function accumulate(energy)
  
  local h = energy:size(1)
  local w = energy:size(2)
  
  -- initialize top row of accumulated energy to energy
  local aEnergy = energy:clone()  
  local aWay = torch.zeros(h, w)
  
  -- go from top row down, computing best accumulated energy in every position 
  for nrow = 2,h do
    
    -- nonstrict monotonicity - we can never go right
    for ncol = 1,w do
      local bestEnergy, bestInd = torch.max(aEnergy[{{nrow-1},{1,ncol}}],2)
      aEnergy[{{nrow},{ncol}}] = bestEnergy + energy[{{nrow},{ncol}}]
      aWay[{{nrow},{ncol}}] = bestInd;
    end
    
  end
  
  return aEnergy, aWay
end

E = torch.rand(5,5)
aE, aWay = accumulate(E)
trj = trace(aE, aWay)

gnuplot.figure(1)
gnuplot.imagesc(E,'color')
gnuplot.figure(2)
gnuplot.imagesc(aE,'color')
