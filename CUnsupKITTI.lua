  local unsupKITTI = torch.class('unsupKITTI', 'unsupSet')

  function unsupKITTI:__init( img1_arr, img2_arr, hpatch, disp_max)
    
    -- img1_arr, img2_arr - N x img_h x img_w tensors of stereo images
    -- hpatch - half patch size ((2*hpatch+1) is size of patch) 
    -- disp_max - maximum disparity value (disparity is shift to the left on img2) 
    
    self.img1_arr = img1_arr;
    self.img2_arr = img2_arr;

    self.hpatch = hpatch; 
    self.disp_max = disp_max;
    self.rowDiffMin = hpatch;

  end
  
  function unsupKITTI:get( batch_size )
    
    local height = self.img1_arr:size(2);
    local width = self.img1_arr:size(3);
    local nbImg = self.img1_arr:size(1)
        
    -- get random image
    local nImg = (torch.random() % (nbImg - 1)) + 1
    
    local epiRef = torch.DoubleTensor(batch_size, 2*self.hpatch+1, width) 
    local epiPos = torch.DoubleTensor(batch_size, 2*self.hpatch+1, width) 
    local epiNeg = torch.DoubleTensor(batch_size, 2*self.hpatch+1, width) 
    
    for nEpi = 1, batch_size do
      
      -- get random epipolar line
      local rowRefCent = torch.random(self.hpatch+1, height-self.hpatch)
      local rowMax = rowRefCent + self.hpatch; 
      local rowMin = rowRefCent - self.hpatch;
      epiRef[nEpi] =  self.img1_arr[{{nImg},{rowMin,rowMax},{}}]:double();
      epiPos[nEpi] =  self.img2_arr[{{nImg},{rowMin,rowMax},{}}]:double();
      
      -- get random negative epipolar line
      local rowNegCent 
      repeat
        rowNegCent = torch.random(self.hpatch+1, height-self.hpatch)
        local rowDif = math.abs(rowNegCent-rowRefCent);
      until (rowDif > self.rowDiffMin) 
      rowMax = rowNegCent + self.hpatch; 
      rowMin = rowNegCent - self.hpatch;
      epiNeg[nEpi] = self.img1_arr[{{nImg},{rowMin,rowMax},{}}]:double():squeeze();
      
    end
       
    return {epiRef, epiPos, epiNeg}, width, self.disp_max 
       
  end