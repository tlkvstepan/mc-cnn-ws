require 'libdprog'
require 'gnuplot'
require 'image'
require 'nn'
utils = dofile('utils.lua')
  
distMin = 2;  
local input = image.load('dprog_test.png',1,'byte'):float()

local E = input:clone():float()
local E_masked = input:clone():float()
local path = input:clone():zero():float()
local pathNonOcc = input:clone():zero():float()
local dim = input:size(1)

local aE = input:clone():zero():float()
local aS = input:clone():zero():float()

dprog.compute(E, path, aE, aS)
dprog.findNonoccPath(path, pathNonOcc)
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
