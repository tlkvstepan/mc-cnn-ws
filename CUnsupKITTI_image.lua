  local unsupKITTI_image = torch.class('unsupKITTI_image', 'unsupSet')
  
  function unsupKITTI_image:__init(baseFolder, setName, hpatch)
    
    self.baseFolder = baseFolder;
    self.nbFrames = 20
    self.maxBatchSize = 370 - hpatch*2
    self.setName = setName 
    self.hpatch = hpatch
    
    if setName == 'kitti' then
        
        self.im0_fname = baseFolder .. '/%s/image_0/%06d_%02d.png' 
        self.im1_fname = baseFolder .. '/%s/image_1/%06d_%02d.png'
        
        self.test_nbIm  = 195
        self.train_nbIm = 194
        self.disp_max = 250      -- guessed from training data
                
    elseif setName == 'kitti15' then
            
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
  
    
  function unsupKITTI_image:subset( prc )
    
    self.nbImActive = math.ceil(prc*self.nbIm)
    
    if( self.nbImActive == 1 )then 
      error('Subset is too small')
    end
    
    self.imActive = torch.randperm(self.nbIm)-1
    self.imActive = self.imActive[{{1,self.nbImActive}}] 
  
  end
    
  function unsupKITTI_image:get( batch_size )
  
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
      im0_fname = self.im0_fname:format(set, nImg, nFrame)  
      im1_fname = self.im1_fname:format(set, nImg, nFrame)
      
    
    until ( utils.file_exists(im0_fname) and utils.file_exists(im1_fname) )
      
      -- read image
    local  im0 = image.load(im0_fname):double()
    local  im1 = image.load(im1_fname):double()
          
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
    
    return {im0, im1}, height, width, self.disp_max 
       
  end