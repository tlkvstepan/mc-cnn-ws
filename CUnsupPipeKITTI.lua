--[[ 
  Function creates training data for training through pipeline.
]]--

local unsupPipeKITTI = torch.class('unsupPipeKITTI')
  
  function unsupPipeKITTI:__init(baseFolder, setName, hpatch)
    
    self.baseFolder = baseFolder;
    
    if( setName == 'kitti_ext' or setName == 'kitti2015_ext' ) then
      self.nbFrames = 20
    else
      self.nbFrames = 1
    end
    
    self.maxBatchSize = 370 - hpatch*2
    self.setName = setName 
    self.hpatch = hpatch
   -- self.height = 370
   -- self.width = 1242;
    self.nb_epi_per_image = 64; -- to use all disparity 
        
    if setName == 'kitti' or setName == 'kitti_ext' then
        
        self.im0_fname = baseFolder .. '/%s/image_0/%06d_%02d.png' 
        self.im1_fname = baseFolder .. '/%s/image_1/%06d_%02d.png'
        
        self.test_nbIm  = 195
        self.train_nbIm = 194
        self.disp_max = 228      -- guessed from training data
        self.nb_ch = 1;
        
    elseif setName == 'kitti2015' or setName == 'kitti2015_ext' then
            
        self.im0_fname = baseFolder .. '/%s/image_2/%06d_%01d.png' 
        self.im1_fname = baseFolder .. '/%s/image_3/%06d_%01d.png'
        self.test_nbIm  = 200
        self.train_nbIm = 200
        self.disp_max = 230
        self.nb_ch = 3;
        
    else
        
        error('SetName should be equal to "kitti" or "kitti15"\n')
        
    end
  
    self.nbIm = (self.test_nbIm + self.train_nbIm)*self.nbFrames     
     
  end

    
  function unsupPipeKITTI:get( batch_size, net_fname )
      
    local nFrame, nImg, set, im0_fname, im1_fname, im1, im0, disp, actual_height, actual_width
    
    local tabWidth = {}
    local tabDisp = {}
    local epiRef = {}
    local epiPos = {}
    local epiMatchIdx = {}
   
    
    for nEpi = 1, batch_size do
      
      tabDisp[nEpi]   = self.disp_max
      
      if( (nEpi-1) % self.nb_epi_per_image == 0 ) then
        
        repeat
        
          -- get image 
          local n = (torch.random() % (self.nbIm)) + 1;
       --   print(n)

          if self.setName == 'kitti2015' or self.setName == 'kitti' then
            nFrame = 10
            nImg = n;
          else
            nFrame = n % self.nbFrames
            nImg = n - nFrame*self.nbFrames;
          end 

          -- test or train?
          if nImg < self.train_nbIm then
            set = 'training' 
            nImg = nImg 
          else
            set = 'testing'
            nImg = nImg - self.train_nbIm;
          end

          -- images file name
          im0_fname = self.im0_fname:format(set, nImg, nFrame)  
          im1_fname = self.im1_fname:format(set, nImg, nFrame)

        until ( utils.file_exists(im0_fname) and utils.file_exists(im1_fname) )
      
        -- run pipeline for the image 
        do
          local set_name
          if self.setName == 'kitti_ext' then
            set_name = 'kitti'
          elseif self.setName == 'kitti2015_ext' then
            set_name = 'kitti2015'
          else 
            set_name = self.setName
          end
          local patt_str = './main.lua %s our-fast -a predict -net_fname %s -left ../mil-mc-cnn/%s -right ../mil-mc-cnn/%s -disp_max %i'
          local exec_str = patt_str:format(set_name, net_fname, im0_fname, im1_fname, torch.round(self.disp_max))
          
          lfs.chdir('../mc-cnn')      -- switch current directory
          local handle = io.popen(exec_str)
          handle :read('*all')
          handle :close()
          lfs.chdir('../mil-mc-cnn')  
        end
        
        
        -- read images
        im0 = image.loadPNG(im0_fname, self.nb_ch, 'byte'):float()
        im1 = image.loadPNG(im1_fname, self.nb_ch, 'byte'):float()
        actual_width  = im0:size(3)
        actual_height = im0:size(2)
        
        -- read disparity
        disp = torch.FloatTensor(torch.FloatStorage('../mc-cnn/disp.bin')):view(actual_height, actual_width);
        ---image.save('disp.png', utils.scale2_01(disp))
        -- read occlusions    
        local occ = torch.FloatTensor(torch.FloatStorage('../mc-cnn/occ.bin')):view(actual_height, actual_width);
        disp[occ:ne( 0 ):byte()] = 1/0
        
        -- convert to luminance
        if self.setName == 'kitti2015' or self.setName == 'kitti2015_ext'  then
          im0 = image.rgb2y(im0)
          im1 = image.rgb2y(im1)
        end

        -- normalize intensity
        im0:add(-im0:mean()):div(im0:std())
        im1:add(-im1:mean()):div(im1:std())
          
    end
   
    tabWidth[nEpi]  = actual_width
   
    -- get random epipolar line
    local rowRefCent = torch.random(self.hpatch+1, actual_height-self.hpatch)
    local rowMax = rowRefCent + self.hpatch; 
    local rowMin = rowRefCent - self.hpatch;
    epiRef[nEpi] =  im0[{{},{rowMin,rowMax},{}}]:double();
    epiPos[nEpi] =  im1[{{},{rowMin,rowMax},{}}]:double();
    local epi_disp =  torch.round(disp[{rowRefCent,{}}]:double());
    
   -- if disp:size(2) ~= im0:size(3) then
   --    error('different width')
   -- end
        
    -- get gt disparity for patches that have match
    -- (col0 - disp) >= hpatch + 1  ==> col1 >= hpatch + 1
    -- col0 <= width - hpatch       
    epi_disp = epi_disp[{{self.hpatch+1, actual_width-self.hpatch}}]
    local col0 = torch.range(1, actual_width-2*self.hpatch)
    
    local col1 = col0 - epi_disp 
    local valid = col1:ge( 1 )
    local nonValid = valid:eq(0)
    col1[nonValid] = 1/0
    
    epiMatchIdx[nEpi] = col1  
     
    --if ( epiPos[nEpi]:size(3)-2*self.hpatch ~= epiMatchIdx[nEpi]:size(1) ) then
    --  d = 1
    --end
     --if epiPos[nEpi]:size(3)-2*self.hpatch ~= epiMatchIdx[nEpi]:size(1) then
     --   k = 1
     --end
      
     
  end
   
  return {epiRef, epiPos, epiMatchIdx}, tabWidth, tabDisp 
       
end