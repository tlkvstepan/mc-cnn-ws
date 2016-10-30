  local unsupKITTI_HD = torch.class('unsupKITTI_HD', 'unsupSet')
  
  function unsupKITTI_HD:__init(baseFolder, setName, hpatch)
    
    self.baseFolder = baseFolder;
    
    if( setName == 'kitti15_ext' or setName == 'kitti15_ext' ) then
      self.nbFrames = 20
    else
      self.nbFrames = 1
    end
    
    self.maxBatchSize = 370 - hpatch*2
    self.setName = setName 
    self.hpatch = hpatch
    --self.img_height = 350;
    
    if setName == 'kitti' or setName == 'kitti_ext' then
        
        self.im0_fname = baseFolder .. '/%s/image_0/%06d_%02d.png' 
        self.im1_fname = baseFolder .. '/%s/image_1/%06d_%02d.png'
        
        self.test_nbIm  = 195
        self.train_nbIm = 194
        self.disp_max = 250      -- guessed from training data
                
    elseif setName == 'kitti15' or setName == 'kitti15_ext' then
            
        self.im0_fname = baseFolder .. '/%s/image_2/%06d_%01d.png' 
        self.im1_fname = baseFolder .. '/%s/image_3/%06d_%01d.png'
        
        self.test_nbIm  = 200
        self.train_nbIm = 200
        self.disp_max = 250
    
    else
        
        error('SetName should be equal to "kitti" or "kitti15"\n')
        
    end
  
    self.nbIm = (self.test_nbIm + self.train_nbIm)*self.nbFrames     
    self:subset(1); 
     
  end
  
    
  function unsupKITTI_HD:subset( prc )
    
    self.nbImActive = math.ceil(prc*self.nbIm)
    
    if( self.nbImActive == 1 )then 
      error('Subset is too small')
    end
    
    self.imActive = torch.randperm(self.nbIm)-1
    self.imActive = self.imActive[{{1,self.nbImActive}}] 
  
  end
    
  function unsupKITTI_HD:get( batch_size )
  
    batch_size = batch_size or self.maxBatchSize
  
    if self.maxBatchSize < batch_size then
      error('Maximum batch size is ' .. self.maxBatchSize .. '\n');
    end
    
    local nFrame, nImg, set, im0_fname, im1_fname
    
    repeat
      
      -- get random image
      local n = self.imActive[torch.random(1,self.nbImActive)] 
      nFrame = n % self.nbFrames
      nImg = torch.floor( n / self.nbFrames ) 
      
      -- test or train?
      if nImg < self.train_nbIm then
        set = 'training' 
        nImg = nImg 
      else
        set = 'testing'
        nImg = nImg - self.train_nbIm;
      end
      
      -- images location
      if self.setName == 'kitti15' or self.setName == 'kitti' then
        nFrame = 10
      end
    
      im0_fname = self.im0_fname:format(set, nImg, nFrame)  
      im1_fname = self.im1_fname:format(set, nImg, nFrame)
      
    
    until ( utils.file_exists(im0_fname) and utils.file_exists(im1_fname) )
      
      -- read image
    local  im0 = image.load(im0_fname):double()
    local  im1 = image.load(im1_fname):double()
        
    img_height = 350    
    local  width  = im0:size(3)
    local  height = im0:size(2)
    im0 = im0:narrow(2, height - img_height + 1, img_height)
    im1 = im1:narrow(2, height - img_height  + 1, img_height)
    
    -- convert to luminance
    if self.setName == 'kitti15' then
      im0 = image.rgb2y(im0)
      im1 = image.rgb2y(im1)
    end

    -- normalize
    im0:add(-im0:mean()):div(im0:std())
    im1:add(-im1:mean()):div(im1:std())

    -- compute dimensions
    local  width  = im0:size(3)
    local  height = im0:size(2)
    
    -- allocate arrays for batch
    local epiRef = torch.DoubleTensor(batch_size, 2*self.hpatch+1, width) 
    local epiPos = torch.DoubleTensor(batch_size, 2*self.hpatch+1, width) 
    local epiNeg = torch.DoubleTensor(batch_size, 2*self.hpatch+1, width) 
    
    for nEpi = 1, batch_size do
      
      -- get random epipolar line
      local rowRefCent = torch.random(self.hpatch+1, height-self.hpatch)
      local rowMax = rowRefCent + self.hpatch; 
      local rowMin = rowRefCent - self.hpatch;
      epiRef[nEpi] =  im0[{{},{rowMin,rowMax},{}}]:double();
      epiPos[nEpi] =  im1[{{},{rowMin,rowMax},{}}]:double();
      
      -- get random negative epipolar line
      local rowNegCent 
      repeat
        rowNegCent = torch.random(self.hpatch+1, height-self.hpatch)
        local rowDif = math.abs(rowNegCent-rowRefCent);
      until (rowDif > self.hpatch) 
      rowMax = rowNegCent + self.hpatch; 
      rowMin = rowNegCent - self.hpatch;
      epiNeg[nEpi] = im1[{{},{rowMin,rowMax},{}}]:double():squeeze();
      
    end
       
    return {epiRef, epiPos, epiNeg}, width, self.disp_max 
       
  end