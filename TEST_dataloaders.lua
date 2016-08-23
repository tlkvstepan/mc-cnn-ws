--[[

Here we test dataloaders

--]]
require 'gnuplot'

dl = require 'dataload'
dofile('CUnsup3EpiSet.lua');
dofile('CUnsup1Patch2EpiSet.lua');
dofile('CSup3PatchSet.lua');

utils = dofile('utils.lua')

img1_arr = torch.squeeze(utils.fromfile('x0.bin')):double();
img2_arr = torch.squeeze(utils.fromfile('x1.bin')):double();
disp_arr = torch.round(torch.squeeze(utils.fromfile('dispnoc.bin'))):double();
img1_arr = img1_arr[{{1,3},{},{}}]
img2_arr = img2_arr[{{1,3},{},{}}]
disp_arr = disp_arr[{{1,3},{},{}}]

-- some parameters
hpatch = 5;
disp_max = torch.round(disp_arr:max())

-- **Test unsupervised 3 epipolar line dataloader**
if false then
  
local set = dl.unsup3EpiSet(img1_arr, img2_arr, hpatch, disp_max);

-- try shuffling
--set:shuffle()

-- split
--local set1, set2 = set:split(0.1)

-- try to get set by indices
inputs, targets = set:index(torch.Tensor{1,2,8,9})
utils.save_tensor('work/ref_epi.jpg', inputs[1][{{1},{},{}}]);
utils.save_tensor('work/pos_epi.jpg', inputs[2][{{1},{},{}}]);
utils.save_tensor('work/neg_epi.jpg', inputs[3][{{1},{},{}}]);

-- test iterator
i = 0;
for k, inputs, targets in set:sampleiter(2, 6) do
   i = i + 1
   print(string.format("batch %d, nsampled = %d", i, k))
end

end


-- **Test unsupervised 1 patch 3 epipolar line dataloader**
if false then
  
local set = dl.unsup1Patch2EpiSet(img1_arr, img2_arr, hpatch, disp_max);

-- try shuffling
set:shuffle()

-- split
local set1, set2 = set:split(0.1)

-- try to get set by indices
inputs, targets = set:index(torch.Tensor{1,2,1000,132})
utils.save_tensor('work/ref_epi.jpg', inputs[1][{{1},{},{}}]);
utils.save_tensor('work/pos_epi.jpg', inputs[2][{{1},{},{}}]);
utils.save_tensor('work/neg_epi.jpg', inputs[3][{{1},{},{}}]);

-- test iterator
i = 0;
for k, inputs, targets in set:sampleiter(2, 6) do
   i = i + 1
   print(string.format("batch %d, nsampled = %d", i, k))
end

end

if true then
  
local set = dl.sup3PatchSet(img1_arr, img2_arr, disp_arr, hpatch);

-- try shuffling
set:shuffle()

-- split
local set1, set2 = set:split(0.1)

-- try to get set by indices
inputs, targets = set:index(torch.range(1,5000))
utils.save_tensor('work/ref_patch.jpg', inputs[1][{{1},{},{}}]);
utils.save_tensor('work/pos_patchi.jpg', inputs[2][{{1},{},{}}]);
utils.save_tensor('work/neg_patch.jpg', inputs[3][{{1},{},{}}]);

-- test iterator
i = 0;
for k, inputs, targets in set:sampleiter(2, 6) do
   i = i + 1
   print(string.format("batch %d, nsampled = %d", i, k))
end

end
