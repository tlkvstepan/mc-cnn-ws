require 'nn'
require 'gnuplot'
gnuplot.setterm('x11')

mcCnnFst = dofile('CBaseNet.lua')
netWrapper = dofile('CNetWrapper.lua')
testFun = dofile('CTestFun.lua')

dofile('CAddMatrix.lua')
dofile('DataLoader.lua');
dofile('CSup2EpiSet.lua');
utils = dofile('utils.lua')

local img1_arr = torch.squeeze(utils.fromfile('data/KITTI12/x0.bin')):float();
local img2_arr = torch.squeeze(utils.fromfile('data/KITTI12/x1.bin')):float();
local disp_arr = torch.round(torch.squeeze(utils.fromfile('data/KITTI12/dispnoc.bin'))):float();
local hpatch = 4
local disp_max = disp_arr:max()
local img_w = img1_arr:size(3);

local set = sup2EpiSet(img1_arr[{{1,194},{},{}}], img2_arr[{{1,194},{},{}}], disp_arr[{{1,194},{},{}}], hpatch);
set:shuffle()

net = torch.load('fnet_2016_09_22_13:39:10_contrast-dprog.t7', 'ascii')
distNet = netWrapper.getDistNet(img_w, disp_max, hpatch, net)

input, target = set:index(torch.range(1, 100))

acc_lt3, acc_lt5, errCases = testFun.epiEpiEval(distNet, {input[1], input[2]}, target)