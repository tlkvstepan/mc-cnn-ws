--require 'libdprog'
require 'gnuplot'
require 'image'
require 'nn'
require 'libdp_kbest'  

local E = image.load('test_dp.png'):squeeze():float()
w = E:size(2)
h = E:size(1)
k = 1

local pathE = torch.Tensor(h,w,k):zero():float()
local pathLen = torch.Tensor(h,w,k):zero():float()
local traceBack = torch.Tensor(h,w,k):zero():float()
local pathOpt = torch.Tensor(h,w,k):zero():float()

dp_kbest.compute(E, pathOpt, pathE, pathLen, traceBack)
gnuplot.imagesc(pathOpt)

dp_kbest.findNonoccPath(path, pathNonOcc,2)
dp_kbest.maskE(pathNonOcc, E_masked, distMin)

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
