require 'image'
require 'torch'
dofile('CUnsupSet.lua')
dofile('CUnsupKITTI_HD.lua')
utils = dofile('utils.lua');              -- Utils for loading and visualization

unsupSet = unsupKITTI_HD( 'data/kitti_ext' ,'kitti', 4 )
start = os.time()
batchInput, width, disp = unsupSet:get(360)
stop = os.time()
print(os.difftime(stop,start))
x= 10