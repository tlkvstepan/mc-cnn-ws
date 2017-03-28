--require 'libdprog'
require 'gnuplot'
require 'image'
require 'nn'
utils = dofile('utils.lua')
--require 'libdprog'  -- C++ DP

local simMat = image.load('dprog_COST.png',1,'byte'):float()
local aprior = image.load('dprog_APRIORI.png',1,'byte'):float()
aprior[aprior:ne( 0 )] = 1
known_index = aprior:nonzero()

dim = simMat:size(1)
r_max = 3
sigma = 0.8
p_min = 0.1

function computeCost(known_index, dim, r_max, sigma, p_min)
    
    local cost = torch.Tensor(dim, dim):fill(p_min)
    local cost_vec = cost:view(dim*dim)
    
    local col0 = known_index:select(2,2)
    local row0 = known_index:select(2,1) 
    
    for drow = -r_max, r_max do
      local row = drow + row0
      for dcol = -r_max, r_max do
        local col = dcol + col0
        local valid = col:ge( 1 ):cmul( col:le( dim ) ):cmul( row:ge( 1 ) ):cmul( row:le( dim ))
        local index = col[valid] + ( row[valid] - 1 )*dim
        local p_prev = cost_vec:index(1,index) 
        local p_new = math.max( torch.exp( -( ( dcol * dcol ) + ( drow * drow ) ) / ( 2 * sigma * sigma ) ), p_min)
        local p = torch.cmax(p_prev, p_new) 
        cost_vec:indexCopy(1, index, p)
        --print(p)
      end
    end
  
    return cost
end  

computeCost(known_index, dim, r_max, sigma, p_min)



local E = simMat:clone():float()
local path = simMat:clone():zero():float()
local aE = simMat:clone():zero():float()
local aS = simMat:clone():zero():float()
local traceBack = simMat:clone():zero():float()
local dim = simMat:size(1)


dprog.compute(E, path, aE, aS, traceBack)


dprog.compute(E, path, aE, aS, traceBack)
dprog.findNonoccPath(path, pathNonOcc,2)
dprog.maskE(pathNonOcc, E_masked, distMin)

indices = pathNonOcc:nonzero() -- valid matches
rows = indices[{{},{1}}]:clone():squeeze():float():add(-1)
cols = indices[{{},{2}}]:clone():squeeze():float():add(-1)


local tmp = input:clone():zero():float()
dprog.collect(tmp, E[pathNonOcc:byte()]:clone(), cols, rows)
--dprogE = E[pathNonOcc:byte()]:clone()
--rowwiseMaxI = torch.zeros(dim):float()
--rowwiseMaxE = torch.zeros(dim):float()
--colwiseMaxI = torch.zeros(dim):float()
--colwiseMaxE = torch.zeros(dim):float()
--dprog.findMaxForRows(E_masked, rows, rowwiseMaxI, rowwiseMaxE)
--dprog.findMaxForCols(E_masked, cols, colwiseMaxI, colwiseMaxE)


--test = E_masked:clone():zero()
--dprog.collect(test, dprogE, cols, rows)
--dprog.collect(test, rowwiseMaxE, rowwiseMaxI, rows)
--dprog.collect(test, colwiseMaxE, cols, colwiseMaxI)
   
--colMaxE, colMaxI = torch.max(_input, 1)
--pathColMax = _input:clone():zero():scatter(1, colMaxI, 1):byte() 
--pathColMax = pathColMax:cmul(torch.repeatTensor(pathNonOcc:sum(1):gt(0),dim,1)) 



--gnuplot.figure()
--gnuplot.imagesc(test,'color')
gnuplot.figure()
gnuplot.imagesc(tmp,'color')
gnuplot.figure()
gnuplot.imagesc(E,'color')
gnuplot.figure()
gnuplot.imagesc(E_masked,'color')
gnuplot.figure()
gnuplot.imagesc(path,'color')
gnuplot.figure()
gnuplot.imagesc(pathNonOcc,'color')
gnuplot.figure()
