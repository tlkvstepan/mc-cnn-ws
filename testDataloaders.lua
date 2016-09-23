--[[

Here we test dataloaders

--]]
require 'gnuplot'
require 'image'
dofile('DataLoader.lua');
dofile('CSup2EpiSet.lua');
utils = dofile('utils.lua')

local img1_arr = torch.squeeze(utils.fromfile('data/KITTI12/x0.bin')):float();
local img2_arr = torch.squeeze(utils.fromfile('data/KITTI12/x1.bin')):float();
local disp_arr = torch.round(torch.squeeze(utils.fromfile('data/KITTI12/dispnoc.bin'))):float();
 hpatch = 4
 
local set = sup2EpiSet(img1_arr[{{1,100},{},{}}], img2_arr[{{1,100},{},{}}], disp_arr[{{1,100},{},{}}], hpatch);
set:shuffle()

for k, inputs, targets in set:sampleiter(1, 30) do
  epiRef, epiPos = unpack(inputs)
  
  
  epiRef = utils.scale2_01(epiRef)
  epiPos = utils.scale2_01(epiPos)
               
  image.save('epiRef.png',epiRef)
  image.save('epiPos.png',epiPos)
  
end

