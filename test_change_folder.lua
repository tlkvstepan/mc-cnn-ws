require 'lfs'


net_fname = '"/HDD1/Data/MIL-MC-CNN/contrast-dprog/fnet_2016_10_15_18:25:25_contrast-dprog.t7"'
local str = './main.lua mb census -a test_te' -- -net_fname '  .. net_fname ..  ' -sm_terminate cnn'

-- run for validation subset and get :error
lfs.chdir('../mc-cnn')
local handle = io.popen(str)
local result = handle:read("*a")
local str_err = string.gsub(result,'\n','');
local test_err = tonumber(str_cost);
