require 'nn'
require 'cunn'

mcCnnFst = dofile('CMcCnnFst.lua')
testFun = dofile('CTestFun.lua')

dofile('DataLoader.lua');
dofile('CSup1Patch1EpiSet.lua');
utils = dofile('utils.lua')

local img1_arr = torch.squeeze(utils.fromfile('data/KITTI12/x0.bin')):float();
local img2_arr = torch.squeeze(utils.fromfile('data/KITTI12/x1.bin')):float();
local disp_arr = torch.round(torch.squeeze(utils.fromfile('data/KITTI12/dispnoc.bin'))):float();
 hpatch = 4
 
local set = sup1Patch1EpiSet(img1_arr[{{1,3},{},{}}], img2_arr[{{1,3},{},{}}], disp_arr[{{1,3},{},{}}], hpatch);
set:shuffle()

fnet, hpatch = mcCnnFst.get(4, 64, 3)
  
input, target = set:index(torch.range(1, 1000))

acc_lt3, acc_lt5, errCases = testFun.epiEval(fnet:cuda(), {input[1]:cuda(), input[2]:cuda()}, target:cuda())