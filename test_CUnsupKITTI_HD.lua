require 'image'
require 'torch'
dofile('CUnsupSet.lua')
dofile('CUnsupKITTI_HD.lua')
utils = dofile('utils.lua');              -- Utils for loading and visualization

unsupSet = unsupKITTI_HD( 'data/kitti15_ext' ,'kitti15', 4 )
unsupSet:subset(0.1)
batchInput, width, disp = unsupSet:get(256)

x= 10