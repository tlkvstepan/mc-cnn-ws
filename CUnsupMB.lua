--[[
  
  Given number of stereo images **CLASS** provides access to dataset that 
  consists of epipolar stripes for one image     
  
  _posEpi_, _negEpi_ are nb_stripes x (2*hpatch+1) x img_w tensors,
  _refEpi_ are nb_stripes x (2*hpatch+1) x (img_w - max_disp) tensors. 
  
  _posEpi_ is conjugate epipolar line for _refEpi_ epipolar line.
  _negEpi_ is not conjugate epipolar line for _refEpi_.
 
]]--

  local unsupMB = torch.class('unsupMB')

  function unsupMB:__init( imgTab, metadata, hpatch)
    
    self.d_exp = 0.2
    self.d_light = 0.2
    self.nb_train = 66 
    
    -- img_tab - table of images
    -- img_tab[nimg][nlight][{nexp,1}]    
    self.imgTab = imgTab;
    
    -- metadata - table of metadata
    -- metadata[nimg][1] - height
    -- metadata[nimg][2] - width
    -- metadata[nimg][3] - disp_max
    self.metadata = metadata
    
    -- halfpatch
    self.hpatch = hpatch; 
    self.rowDiffMin = hpatch;
    
    self.nb_epi_per_image = 10
    
  end

  function unsupMB:get(batch_size)
    
    -- function returns random batch from dataset
    -- for efficiency, in single batch all epipolar 
    -- are taken from same image (disparity range and
    -- width is same)
    
    local widthTab = {}
    local dispTab = {}
    local epiRef = {} 
    local epiPos = {} 
    local epiNeg = {} 
   
    local img0 
    local img1
    local height, width, disp 
    
    for nEpi = 1, batch_size do
    
      if( (nEpi-1) % self.nb_epi_per_image == 0 ) then
      
        local nbImg = #self.imgTab
        
        -- get random image
        local nImg = (torch.random() % (nbImg - 1)) + 1
        local nbLight = #self.imgTab[nImg]
            
        height = self.metadata[{nImg,1}] 
        width = self.metadata[{nImg,2}]
        disp = self.metadata[{nImg,3}]
    
        if nImg <= 60 then
          -- if belongs to training set
        
          -- get random lighting (but not too far from center)
          -- only for >=2 there is left image
          local nLight0 = (torch.random() % (nbLight - 1)) + 2
          local nbExp = self.imgTab[nImg][nLight0]:size(1)
          
          -- get random exposure
          local nExp0 = (torch.random() % nbExp) + 1
          
          local nLight1 = nLight0
          local nExp1 = nExp0
          
          -- get different exposure for second image
          -- with some random probability
          if torch.uniform() < self.d_light then
            nLight1 = math.max(2, nLight0 - 1)
          end
          
          if torch.uniform() < self.d_exp then
            nExp1 = (torch.random() % self.imgTab[nImg][nLight1]:size(1)) + 1
          end
        
          img0 = self.imgTab[nImg][nLight0][{nExp0,1}]
          img1 = self.imgTab[nImg][nLight1][{nExp1,2}]
        
        else
          
          img0 = self.imgTab[nImg][1][1]
          img1 = self.imgTab[nImg][1][2]
        
        end
      
      end
      
      widthTab[nEpi] = width
      dispTab[nEpi] = disp
      
      -- get random epipolar lines
      local rowRefCent = torch.random(self.hpatch+1, height-self.hpatch)
      local rowMax = rowRefCent + self.hpatch; 
      local rowMin = rowRefCent - self.hpatch;
      epiRef[nEpi] = img0[{{},{rowMin,rowMax},{}}]:double();
      epiPos[nEpi] = img1[{{},{rowMin,rowMax},{}}]:double();
      
      -- get random negative epipolar line
      local rowNegCent 
      
      repeat
        rowNegCent = torch.random(self.hpatch+1, height-self.hpatch)
        local rowDif = math.abs(rowNegCent-rowRefCent);
      until (rowDif > self.rowDiffMin) 
      
      rowMax = rowNegCent + self.hpatch; 
      rowMin = rowNegCent - self.hpatch;
      epiNeg[nEpi] = img1[{{},{rowMin,rowMax},{}}]:double();
      
    end
       
    return {epiRef, epiPos, epiNeg}, widthTab, dispTab
    
  end 
    




