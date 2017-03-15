require 'gnuplot'
require 'image'
require 'torch'
require 'lfs'
require 'os'

torch.manualSeed(1)

utils = dofile('utils.lua');              
dofile('CUnsupPipeMB_with_GT.lua');

local metadata_fname = 'data/mb/meta.bin'
local metadata = utils.fromfile(metadata_fname)
local img_tab = {}
local disp_tab = {}
for n = 1,metadata:size(1) do
  local img_light_tab = {}
  light = 1
  while true do
    fname = ('data/mb/x_%d_%d.bin'):format(n, light)
    if not paths.filep(fname) then
      break
    end
    table.insert(img_light_tab, utils.fromfile(fname))
    light = light + 1
  end
  table.insert(img_tab, img_light_tab)
  fname = ('data/mb/dispnoc%d.bin'):format(n)
  if paths.filep(fname) then
    table.insert(disp_tab, utils.fromfile(fname))
  end
  if metadata[{n,3}] == -1 then -- fill max_disp for train set
    metadata[{n,3}] = disp_tab[n]:max()
  end
end


hpatch = 4
batch_size = 128
unique_name = 'test_mb_pipe'
net_fname = '/HDD1/Dropbox/Research/01_code/mil-mc-cnn/work/TRAIN_CONTRASTIVEDP_FSTXXL_KITTIEXT/metricNet_TRAIN_CONTRASTIVEDP_FSTXXL_KITTIEXT.t7'

unsupSet = unsupPipeMB_with_GT(img_tab, metadata, disp_tab, hpatch, unique_name);
_TR_INPUT_, _WIDTH_TAB_, _DISP_MAX_TAB_  = unsupSet:get( batch_size, net_fname )



