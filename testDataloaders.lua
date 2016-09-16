--[[

Here we test dataloaders

--]]
require 'gnuplot'
require 'image'
dofile('DataLoader.lua');
dofile('CSup1Patch1EpiSet.lua');
utils = dofile('utils.lua')

local img1_arr = torch.squeeze(utils.fromfile('data/KITTI12/x0.bin')):float();
local img2_arr = torch.squeeze(utils.fromfile('data/KITTI12/x1.bin')):float();
local disp_arr = torch.round(torch.squeeze(utils.fromfile('data/KITTI12/dispnoc.bin'))):float();
 hpatch = 4
 
local set = sup1Patch1EpiSet(img1_arr[{{1,3},{},{}}], img2_arr[{{1,3},{},{}}], disp_arr[{{1,3},{},{}}], hpatch);
set:shuffle()

for k, inputs, targets in set:sampleiter(1, 30) do
  patch, epi = unpack(inputs)
  epi = epi[{{},{},{hpatch, -(hpatch+1)}}];
  
  epi_red = epi:clone(); 
  epi_red[{{},{},{targets:squeeze()-hpatch, targets:squeeze()+hpatch}}] = epi_red[{{},{},{targets:squeeze()-hpatch, targets:squeeze()+hpatch}}] + patch
  
  epi = utils.scale2_01(epi)
  epi_red = utils.scale2_01(epi_red)
  im = torch.cat({epi, epi_red, epi}, 1)
               
  image.save('epi.png',im)
end

