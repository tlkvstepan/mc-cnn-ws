--[[
  
  Given number of stereo images **CLASS** provides access to dataset that 
  consists of epipolar stripes for one image     
  
  _posEpi_, _negEpi_ are nb_stripes x (2*hpatch+1) x img_w tensors,
  _refEpi_ are nb_stripes x (2*hpatch+1) x (img_w - max_disp) tensors. 
  
  _posEpi_ is conjugate epipolar line for _refEpi_ epipolar line.
  _negEpi_ is not conjugate epipolar line for _refEpi_.
 
]]--

  local unsupPipeMB = torch.class('unsupPipeMB')

  function unsupPipeMB:__init( imgTab, metadata, hpatch, unique_name)
    
    self.d_exp = 0.2
    self.d_light = 0.2
    self.nb_train = 66 
    
    self.occ_fname = 'occ_' .. unique_name
    self.disp_fname = 'disp_' .. unique_name
    self.right_fname = 'right_' .. unique_name
    self.left_fname = 'left_' .. unique_name
        
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
    
    self.nb_epi_per_image = 100
    
  end

  function unsupPipeMB:get(batch_size, net_fname)
    
    -- function returns random batch from dataset
    -- for efficiency, in single batch all epipolar 
    -- are taken from same image (disparity range and
    -- width is same)
    
    local widthTab = {}
    local dispTab = {}
    local epiRef = {} 
    local epiPos = {} 
    local epiMatchIdx = {} 
   
    local img0 
    local img1
    local height, width, disp_max, disp, occ
    
    for nEpi = 1, batch_size do
    
      if( (nEpi-1) % self.nb_epi_per_image == 0 ) then
      
        repeat
          local nbImg = #self.imgTab
          
          -- get random image
          local nImg = (torch.random() % (nbImg - 1)) + 1
          local nbLight = #self.imgTab[nImg]
              
          height = self.metadata[{nImg,1}] 
          width = self.metadata[{nImg,2}]
          disp_max = self.metadata[{nImg,3}]
      
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
                 
          -- save images 
          image.save(self.left_fname ..'.png', utils.scale2_01(img0))
          image.save(self.right_fname ..'.png', utils.scale2_01(img1))
          
          --in case if file writing is async
          repeat 
          until utils.file_exists(self.right_fname ..'.png') and utils.file_exists(self.left_fname ..'.png')
          
          -- compute 
          local patt_str = './main.lua mb our-fast -a predict -net_fname %s  -disp %s.bin -occ %s.bin  -left ../mil-mc-cnn/%s -right ../mil-mc-cnn/%s -disp_max %i'
          local exec_str = patt_str:format(net_fname, self.disp_fname, self.occ_fname, self.left_fname ..'.png', self.right_fname ..'.png', torch.round(disp_max))
          
          lfs.chdir('../mc-cnn')      -- switch current directory
          print('computing pipeline results\n');
          local handle = io.popen(exec_str)
          handle :read('*all')
          handle :close()
          lfs.chdir('../mil-mc-cnn')  
        
        --in case if main fails
        until utils.file_exists('../mc-cnn/' .. self.disp_fname.. '.bin') and utils.file_exists('../mc-cnn/' .. self.occ_fname.. '.bin')              
        
        disp = torch.FloatTensor(torch.FloatStorage('../mc-cnn/' .. self.disp_fname.. '.bin')):view(height, width);
        os.remove('../mc-cnn/' .. self.disp_fname.. '.bin')
        image.save(self.disp_fname ..'.png', utils.scale2_01(disp))
        occ = torch.FloatTensor(torch.FloatStorage('../mc-cnn/' .. self.occ_fname.. '.bin')):view(height, width);
        os.remove('../mc-cnn/' .. self.occ_fname.. '.bin')
          
        image.save(self.occ_fname ..'.png', utils.scale2_01(occ))
        disp[occ:ne( 0 ):byte()] = 1/0
        
       -- print(disp:size())
        --print(occ:size())
        --print(img0:size())
        --print(img1:size())
        
      end
      
      
      widthTab[nEpi] = width
      dispTab[nEpi] = disp_max
      
      -- get random epipolar lines
      local rowRefCent = torch.random(self.hpatch+1, height-self.hpatch)
      local rowMax = rowRefCent + self.hpatch; 
      local rowMin = rowRefCent - self.hpatch;
      epiRef[nEpi] = img0[{{},{rowMin,rowMax},{}}]:double();
      epiPos[nEpi] = img1[{{},{rowMin,rowMax},{}}]:double();
      local epi_disp =  torch.round(disp[{rowRefCent,{}}]:double());
        
      epi_disp = epi_disp[{{self.hpatch+1, width-self.hpatch}}]
      local col0 = torch.range(1, width-2*self.hpatch)
    
      local col1 = col0 - epi_disp 
      local valid = col1:ge( 1 )
      local nonValid = valid:eq(0)
      col1[nonValid] = 1/0
      
      epiMatchIdx[nEpi] = col1;
      
      collectgarbage();
    end
       
    return {epiRef, epiPos, epiMatchIdx}, widthTab, dispTab
    
  end 
    




