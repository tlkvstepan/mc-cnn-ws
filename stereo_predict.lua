
--function stereo_predict(x_batch, id)
--   local vols, vol

--   if arch == 'ad' then
--      vols = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(0 / 0)
--      adcensus.ad(x_batch[{{1}}], x_batch[{{2}}], vols[{{1}}], -1)
--      adcensus.ad(x_batch[{{2}}], x_batch[{{1}}], vols[{{2}}], 1)
--   end

-- --- DENSE SIFT
--   if arch == 'dsift' then
--      vols = torch.Tensor(2, disp_max, x_batch:size(3), x_batch:size(4)):float()
--      x_batch_host = x_batch:float();
--      cv.dsift(x_batch_host[{{1}}], x_batch_host[{{2}}], vols[{{1}}], vols[{{2}}])
--      vols = vols:cuda();
--   end
-----

-- --- DAISY
--   if arch == 'daisy' then
--      vols = torch.Tensor(2, disp_max, x_batch:size(3), x_batch:size(4)):float()
--      x_batch_host = x_batch:float();
--      cv._daisy(x_batch_host[{{1}}], x_batch_host[{{2}}], vols[{{1}}], vols[{{2}}])
--      vols = vols:cuda();
--   end
-----

---- TO-DO restore VGG support
----[[
-- --- VGG
--   if arch == 'vgg' then
--      vols = torch.Tensor(2, disp_max, x_batch:size(3), x_batch:size(4)):float()
--      x_batch_host = x_batch:float();
--      --- TO-DO
--      all_filters = '/HDD1/Programs/opencv-dlco/workspace/filters.h5'
--      filter_weights = '/HDD1/Programs/opencv-dlco/workspace/pr-learn/liberty-0.035-0.250-pr.h5'
--      projection = '/HDD1/Programs/opencv-dlco/workspace/pj-learn/notredame-liberty-0.035-0.250-pr#7-0.0010-0.100-pj.h5'
--      cv.vgg(x_batch_host[{{1}}], x_batch_host[{{2}}], vols[{{1}}], vols[{{2}}], all_filters, filter_weights, projection)
--      vols = vols:cuda();
--   end
-----
--]]--

--   if arch == 'census' then
--      vols = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(0 / 0)
--      adcensus.census(x_batch[{{1}}], x_batch[{{2}}], vols[{{1}}], -1)
--      adcensus.census(x_batch[{{2}}], x_batch[{{1}}], vols[{{2}}], 1)
--   end

--   if arch == 'our'  then
--      forward_free(net_te, x_batch:clone())
--      vols = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(0 / 0)
--      adcensus.StereoJoin(net_te.output[{{1}}], net_te.output[{{2}}], vols[{{1}}], vols[{{2}}])
--      fix_border(net_te, vols[{{1}}], -1)
--      fix_border(net_te, vols[{{2}}], 1)
--      clean_net(net_te)
--   end

  require 'nn'
  require 'cudnn'
  require 'cunn'
  require 'image'
  
  utils = dofile('utils.lua')
  
  dofile('CUnsupSet.lua')     
  dofile('CUnsupMB.lua')    -- MB

  dofile('CUnsupKITTI_image.lua')
 
 
  package.cpath = package.cpath .. ';../mc-cnn/?.so'
  require('libcv')
  require 'libadcensus'
  
  include('../mc-cnn/Margin2.lua') 
  include('../mc-cnn/Normalize2.lua') 
  include('../mc-cnn/BCECriterion2.lua')

  include('../mc-cnn/StereoJoin.lua') 
  include('../mc-cnn/StereoJoin1.lua') 
  include('../mc-cnn/SpatialConvolution1_fw.lua') 
  include('../mc-cnn/SpatialLogSoftMax.lua') 
 
 
  -- parameters
  hpatch = 4
  batch_size = 1
  
  -- load training set
  unsupSet = unsupKITTI_image('data/kitti_ext', 'kitti', hpatch);
  
  -- load feature net
  baseNet = dofile('CBaseNet.lua');         
  fnet, hpatch = baseNet.get(4, 64, 3, true)

  
  -- get input
  input, height, width, disp_max = unsupSet:get(1)
  height = 60
 
  -- cudify
  for i = 1,2 do
    input[i] = image.crop(input[i], 0, 0, width, 60)
    input[i] = input[i]:cuda()
  end
  fnet:cuda()
  cudnn.convert(fnet,cudnn)
  
  -- pass image through network
  output = {}
  for i = 1,2 do
    output[i] = fnet:forward( input[i] )
  end 
  
  vol = {}
  for i = 1, 2 do
    vol[i] = torch.CudaTensor(1, disp_max, height, width):fill(0 / 0)
  end
  adcensus.StereoJoin(output[1], output[2], vol[1], vol[2])
  
  x = 1
--   if arch == 'fast' or arch == 'our'  then
--      forward_free(net_te, x_batch:clone())
--      vols = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(0 / 0)
--      adcensus.StereoJoin(net_te.output[{{1}}], net_te.output[{{2}}], vols[{{1}}], vols[{{2}}])
--      fix_border(net_te, vols[{{1}}], -1)
--      fix_border(net_te, vols[{{2}}], 1)
--      clean_net(net_te)
--   end

--   disp = {}
--   local mb_directions = opt.a == 'predict' and {1, -1} or {-1}
--   for _, direction in ipairs(dataset == 'mb' and mb_directions or {1, -1}) do
--      sm_active = true

--      if arch == 'slow' then
--         if opt.use_cache then
--            vol = torch.load(('cache/%s_%d.t7'):format(id, direction))
--         else
--            local output = forward_free(net_te, x_batch:clone())
--            clean_net(net_te)
--            collectgarbage()

--            vol = torch.CudaTensor(1, disp_max, output:size(3), output:size(4)):fill(0 / 0)
--            collectgarbage()
--            for d = 1,disp_max do
--               local l = output[{{1},{},{},{d,-1}}]
--               local r = output[{{2},{},{},{1,-d}}]
--               x_batch_te2:resize(2, r:size(2), r:size(3), r:size(4))
--               x_batch_te2[1]:copy(l)
--               x_batch_te2[2]:copy(r)
--               x_batch_te2:resize(1, 2 * r:size(2), r:size(3), r:size(4))
--               forward_free(net_te2, x_batch_te2)
--               vol[{1,d,{},direction == -1 and {d,-1} or {1,-d}}]:copy(net_te2.output[{1,1}])
--            end
--            clean_net(net_te2)
--            fix_border(net_te, vol, direction)
--            if opt.make_cache then
--               torch.save(('cache/%s_%d.t7'):format(id, direction), vol)
--            end
--         end
--         collectgarbage()
--      -- TO-DO restore VGG support
--      --     elseif arch == 'fast' or arch == 'ad' or arch == 'dsift' or arch == 'vgg' or arch == 'daisy' or arch == 'census' then
--      elseif arch == 'fast' or arch == 'our' or arch == 'ad' or arch == 'dsift' or arch == 'daisy' or arch == 'census' then
        
--         vol = vols[{{direction == -1 and 1 or 2}}]
--      end
--      sm_active = sm_active and (opt.sm_terminate ~= 'cnn')

--      -- cross computation
--      local x0c, x1c
--      if sm_active and opt.sm_skip ~= 'cbca' then
--         x0c = torch.CudaTensor(1, 4, vol:size(3), vol:size(4))
--         x1c = torch.CudaTensor(1, 4, vol:size(3), vol:size(4))
--         adcensus.cross(x_batch[1], x0c, opt.L1, opt.tau1)
--         adcensus.cross(x_batch[2], x1c, opt.L1, opt.tau1)
--         local tmp_cbca = torch.CudaTensor(1, disp_max, vol:size(3), vol:size(4))
--         for i = 1,opt.cbca_i1 do
--            adcensus.cbca(x0c, x1c, vol, tmp_cbca, direction)
--            vol:copy(tmp_cbca)
--         end
--         tmp_cbca = nil
--         collectgarbage()
--      end
--      sm_active = sm_active and (opt.sm_terminate ~= 'cbca1')

--      if sm_active and opt.sm_skip ~= 'sgm' then
--         vol = vol:transpose(2, 3):transpose(3, 4):clone()
--         collectgarbage()
--         do
--            local out = torch.CudaTensor(1, vol:size(2), vol:size(3), vol:size(4))
--            local tmp = torch.CudaTensor(vol:size(3), vol:size(4))
--            for _ = 1,opt.sgm_i do
--               out:zero()
--               adcensus.sgm2(x_batch[1], x_batch[2], vol, out, tmp, opt.pi1, opt.pi2, opt.tau_so,
--                  opt.alpha1, opt.sgm_q1, opt.sgm_q2, direction)
--               vol:copy(out):div(4)
--            end
--            vol:resize(1, disp_max, x_batch:size(3), x_batch:size(4))
--            vol:copy(out:transpose(3, 4):transpose(2, 3)):div(4)

----            local out = torch.CudaTensor(4, vol:size(2), vol:size(3), vol:size(4))
----            out:zero()
----            adcensus.sgm3(x_batch[1], x_batch[2], vol, out, opt.pi1, opt.pi2, opt.tau_so,
----               opt.alpha1, opt.sgm_q1, opt.sgm_q2, direction)
----            vol:mean(out, 1)
----            vol = vol:transpose(3, 4):transpose(2, 3):clone()
--         end
--         collectgarbage()
--      end
--      sm_active = sm_active and (opt.sm_terminate ~= 'sgm')

--      if sm_active and opt.sm_skip ~= 'cbca' then
--         local tmp_cbca = torch.CudaTensor(1, disp_max, vol:size(3), vol:size(4))
--         for i = 1,opt.cbca_i2 do
--            adcensus.cbca(x0c, x1c, vol, tmp_cbca, direction)
--            vol:copy(tmp_cbca)
--         end
--      end
--      sm_active = sm_active and (opt.sm_terminate ~= 'cbca2')

--      if opt.a == 'predict' then
--         local fname = direction == -1 and 'left' or 'right'
--         print(('Writing %s.bin, %d x %d x %d x %d'):format(fname, vol:size(1), vol:size(2), vol:size(3), vol:size(4)))
--         torch.DiskFile(('%s.bin'):format(fname), 'w'):binary():writeFloat(vol:float():storage())
--         collectgarbage()
--      end

--      _, d = torch.min(vol, 2)
--      disp[direction == 1 and 1 or 2] = d:add(-1)
--   end
--   collectgarbage()

--   if dataset == 'kitti' or dataset == 'kitti2015' then
--      local outlier = torch.CudaTensor():resizeAs(disp[2]):zero()
--      adcensus.outlier_detection(disp[2], disp[1], outlier, disp_max)
--      if sm_active and opt.sm_skip ~= 'occlusion' then
--         disp[2] = adcensus.interpolate_occlusion(disp[2], outlier)
--      end
--      sm_active = sm_active and (opt.sm_terminate ~= 'occlusion')

--      if sm_active and opt.sm_skip ~= 'occlusion' then
--         disp[2] = adcensus.interpolate_mismatch(disp[2], outlier)
--      end
--      sm_active = sm_active and (opt.sm_terminate ~= 'mismatch')
--   end
--   if sm_active and opt.sm_skip ~= 'subpixel_enchancement' then
--      disp[2] = adcensus.subpixel_enchancement(disp[2], vol, disp_max)
--   end
--   sm_active = sm_active and (opt.sm_terminate ~= 'subpixel_enchancement')

--   if sm_active and opt.sm_skip ~= 'median' then
--      disp[2] = adcensus.median2d(disp[2], 5)
--   end
--   sm_active = sm_active and (opt.sm_terminate ~= 'median')

--   if sm_active and opt.sm_skip ~= 'bilateral' then
--      disp[2] = adcensus.mean2d(disp[2], gaussian(opt.blur_sigma):cuda(), opt.blur_t)
--   end

--   return disp[2]
--end