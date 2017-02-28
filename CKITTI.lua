local kitti = torch.class('kitti')
  
  function kitti:__init(baseFolder, setName, hpatch)
    
    self.baseFolder = baseFolder;
    
    if( setName == 'kitti_ext' or setName == 'kitti2015_ext' ) then
      self.nbFrames = 20
    else
      self.nbFrames = 1
    end
    
    self.maxBatchSize = 370 - hpatch*2
    self.setName = setName 
    self.hpatch = hpatch
    self.height = 370
    self.width = 1242;
    self.nb_epi_per_image = 30; 
        
    if setName == 'kitti' or setName == 'kitti_ext' then
        
        self.gt_fname  = baseFolder .. '/training/%06d_%02d.png' 
        self.im0_fname = baseFolder .. '/%s/image_0/%06d_%02d.png' 
        self.im1_fname = baseFolder .. '/%s/image_1/%06d_%02d.png'
        
        self.test_nbIm  = 195
        self.train_nbIm = 194
        self.disp_max = 228      -- guessed from training data
        self.nb_ch = 1;
        
    elseif setName == 'kitti2015' or setName == 'kitti2015_ext' then
        
        self.gt_fname  = baseFolder .. '/training/%06d_%02d.png'
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
  
  function kitti:get( batch_size )
      
   
    local nFrame, nImg, set, gt_fname, im0_fname, im1_fname, gt, im1, im0, actual_height, actual_width
    
    local tabWidth = {}
    local tabDisp = {}
    local epiRef = {}
    local epiPos = {}
    local epiNeg = {}
    local epiGT = {}      -- size is (width - 2 x hpatch)
    
    for nEpi = 1, batch_size do
      
      tabDisp[nEpi]   = self.disp_max
            
      if( (nEpi-1) % self.nb_epi_per_image == 0 ) then
        
       repeat
        
        local n = (torch.random() % (self.nbIm)) + 1;
      --  local n = self.imActive[idx]
      -- self.imActive        
        -- get random image
       --- local n = self.imActive[torch.random(1,self.nbImActive)] 
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
        
        -- images location
        im0_fname = self.im0_fname:format(set, nImg, nFrame)  
        im1_fname = self.im1_fname:format(set, nImg, nFrame)
        gt_fname  = self.gt_fname:format(nImg, nFrame)
        
      until ( utils.file_exists(im0_fname) and utils.file_exists(im1_fname) )
      
      -- read image
      im0 = image.loadPNG(im0_fname, self.nb_ch, 'byte'):float()
      im1 = image.loadPNG(im1_fname, self.nb_ch, 'byte'):float()

      actual_width  = im0:size(3)
      actual_height = im0:size(2)
    
      -- convert to luminance
      if self.setName == 'kitti2015' or self.setName == 'kitti2015_ext'  then
        im0 = image.rgb2y(im0)
        im1 = image.rgb2y(im1)
      end

      -- normalize intensity
      im0:add(-im0:mean()):div(im0:std())
      im1:add(-im1:mean()):div(im1:std())
      
      -- get gt
      if set == 'training' then
      
        gt = image.load(gt_fname, 1, 'float') 
        gt[gt:eq( 0 )] = 1/0
      
      else
        
        gt = torch.Tensor(im0:size()):fill( 1/0 ):squeeze()
        
      end
    
    end
   
    tabWidth[nEpi]  = actual_width
   
    -- get random epipolar line
    local rowRefCent = torch.random(self.hpatch+1, actual_height-self.hpatch)
    local rowMax = rowRefCent + self.hpatch; 
    local rowMin = rowRefCent - self.hpatch;
    epiRef[nEpi] =  im0[{{},{rowMin,rowMax},{}}]:double();
    epiPos[nEpi] =  im1[{{},{rowMin,rowMax},{}}]:double();
    
    -- get random negative epipolar line
    local rowNegCent 
    repeat
      rowNegCent = torch.random(self.hpatch+1, actual_height-self.hpatch)
      local rowDif = math.abs(rowNegCent-rowRefCent);
    until (rowDif > self.hpatch) 
    rowMax = rowNegCent + self.hpatch; 
    rowMin = rowNegCent - self.hpatch;
    epiNeg[nEpi] = im1[{{},{rowMin,rowMax},{}}]:double();
    
    -- get ground truth
    local dispEpi = torch.round( gt[{rowRefCent,{}}]:double() );
    dispEpi = dispEpi[{{self.hpatch+1, actual_width-self.hpatch}}]
    local idx0 = torch.range(1, actual_width-2*self.hpatch)
    local idx1 = idx0 - dispEpi 
    local valid = idx1:ge( 1 )
    local nonValid = valid:eq( 0 )
    idx1[nonValid] = 1/0
    
    epiGT[nEpi] = idx1
    
  end
   
  return {epiRef, epiPos, epiNeg, epiGT}, tabWidth, tabDisp 
       
end