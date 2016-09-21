--[[
  
  Given number of stereo images **CLASS** provides access to dataset that 
  consists of patch, corresponding epipolar line and gt disparity:     
  
  _patch_, _epi_ are nb_patch x (2*hpatch+1) x (2*hpatch+1) tensors,
  _patch_ is reference patch.
  _epi_ is correspoinding epipolar line
 
  _gtDisp_ is nb_patch tensor with ground truth disparities
   
]]--

local sup1Patch1EpiSet = torch.class('sup1Patch1EpiSet', 'DataLoader')
local image = require 'image'

function sup1Patch1EpiSet:__init(img1_arr, img2_arr, disp_arr, hpatch)
   
  disp_arr = torch.round(disp_arr);
  
  self.img1_arr = img1_arr;
  self.img2_arr = img2_arr;
  self.disp_arr = disp_arr; 
   
  self.nb_pairs = img1_arr:size(1);
  self.img_h = img1_arr:size(2);
  self.img_w = img1_arr:size(3);
  self.hpatch = hpatch; 
  self.disp_max = disp_arr:max();
    
  -- initialize list ids of patches with ground truth
  self.id = self:get_valid_id();
  
end


function sup1Patch1EpiSet:pair_col_row_2_id(pair, col, row) 
  
  -- Function converts image pair number, x and y of patch center to patch id
  -- Patches are indexed in row-first order
  
  if type(pair) ~= 'number' then
      pair = pair:double()
  end
  if type(col) ~= 'number' then
    col = col:double()
  end
  if type(row) ~= 'number' then
      row = row:double()
  end 
  
  col = col - 1;
  row = row - 1;
  pair = pair - 1;
  local id = ( (row + pair * self.img_h) * self.img_w + col )
  local id = id + 1;
  
  if type(id) ~= 'number'then
    id = id:double()
  end 

  return id

end

function sup1Patch1EpiSet:id_2_pair_col_row(id)
  
  -- Function convert patch's id to pair number, x and y
  -- Patches are indexed in row-first order
  
  if type(id) ~= 'number' then
    id = id:double()
  end 
  
  id = id - 1;
  
  local area = (self.img_w * self.img_h)
  local pair = ( torch.floor(id / area) );
  local reminder = (id % area);
  local y =  torch.floor(reminder / self.img_w);
  local x = reminder % self.img_w ;
  
  if type(pair) ~=  'number' then
    pair = pair:double()
  end
  if type(x) ~= 'number' then
    x = x:double()
  end

  if type(y) ~= 'number' then
    y = y:double()
  end 


  return (pair + 1), (x + 1), (y + 1);

end
  

function sup1Patch1EpiSet:get_patch(id, img)
  
  -- Function returns patches when given id 
  local pair, col, row =  self:id_2_pair_col_row(id);
  local row_max = row + self.hpatch; 
  local row_min = row - self.hpatch;
  local col_max = col + self.hpatch; 
  local col_min = col - self.hpatch;
  
  local patch = torch.Tensor(id:numel(), 2*self.hpatch+1, 2*self.hpatch+1)
    
  for n = 1,id:numel() do
      -- we keep data in float, but when queried convert to double
      patch[{{n},{},{}}] = torch.squeeze(img[{{pair[n]},{row_min[n],row_max[n]},{col_min[n], col_max[n]}}]:double()); 
  end
  
  return patch
end

function sup1Patch1EpiSet:get_disp(id, disp)
    
    -- Function returns gt disparity, given patchs id
        
    local pair, col, row =  self:id_2_pair_col_row(id);
    local gtDisp = torch.Tensor(id:numel())
    
    for n = 1, id:numel() do
      -- we keep data in float, but when queried convert to double
      gtDisp[n] = torch.squeeze(disp[{{pair[n]},{row[n]},{col[n]}}]:double());
    end
          
    return gtDisp;
  
end

function sup1Patch1EpiSet:get_epi(id, img)
    
    -- Function returns epipolar line, given patchs id
        
    local pair, col, row =  self:id_2_pair_col_row(id);
    local row_max = row + self.hpatch; 
    local row_min = row - self.hpatch;
    
    local epi = torch.Tensor(id:numel(), 2*self.hpatch+1, self.disp_max + 2*self.hpatch + 1)
    
    for n = 1, id:numel() do
      local col_max = col[n] + self.hpatch;
      local col_min = col[n] - self.disp_max - self.hpatch;
      -- we keep data in float, but when queried convert to double 
      epi[{{n},{},{}}] = torch.squeeze(img[{{pair[n]},{row_min[n],row_max[n]},{col_min, col_max}}]:double());
    end
          
    return epi;
  
end

function sup1Patch1EpiSet:get_valid_id()
  --[[ 
       Function make list of valid patch indices 
       Patch is valid if 
       
       1. (x,y) of its center fulfil below requirements
          x in [(disp_max + hpatch + 1), img_w - hpatch] and  
          y in [hpatch + 1, img_h - hpatch]
       
       2. it's disparity is known 
       
  ]]--
  
    -- make list of indices for single image
    local x = torch.range(1, self.img_w);
    x = x:view(1,x:numel()):clone();
    local y = torch.range(1, self.img_h);
    y = y:view(y:numel(),1):clone();
    local xx = torch.repeatTensor(x,y:size(1),1);
    local yy = torch.repeatTensor(y,1,x:size(2));
         
    -- compute indexes of valid patches in all image
    local id = {}
    for npair = 1, self.nb_pairs do
          
      -- get disparity map
      local disp = torch.squeeze(self.disp_arr[{{npair},{},{}}]):double();
          
      -- find points where disp is not zero
      local mask = torch.squeeze(disp:ge(1));
      if mask then 
          
            -- find x and y of these points 
            local cur_xx = xx[mask];
            local cur_yy = yy[mask];
            local cur_disp = disp[mask];
            
            -- find points where we can fit patch in reference and conjugate image
            local mask = cur_xx:ge(self.disp_max + self.hpatch + 1) 
            local mask_ = cur_xx:le(self.img_w-self.hpatch)
            mask:cmul(mask_);
            local mask_ = cur_yy:ge(self.hpatch + 1)
            mask:cmul(mask_);
            local mask_ = cur_yy:le(self.img_h-self.hpatch)
            mask:cmul(mask_);
                        
            -- find x, y for these points
            local cur_xx = cur_xx[mask];
            local cur_yy = cur_yy[mask];
                        
            local cur_ID = self:pair_col_row_2_id(npair, cur_xx, cur_yy);
            
            table.insert(id, cur_ID) 
      end
    end
    
    local id = torch.cat(id, 1);
    return id
end

function sup1Patch1EpiSet:index(indices, inputs, targets)   
   
   --[[ 
      Given array of indices return _inputs_ and _targets_
    
      _inputs_ is table of 2 tensors: _patch_, _epi_  
      _patch_ is nb_patch x (2*hpatch+1) x (2*hpatch+1) tensor,
      _epi_ is nb_patch x (2*hpatch+1) x (2*hpatch + disp_max) tensor
        
      _targets_ is nb_patch tensor of gt disparity values
    ]]--  

    local nb_indices = indices:numel()
    local id = self.id:index(1, indices:long())
    
    local patch = self:get_patch(id, self.img1_arr)
    local epi = self:get_epi(id, self.img2_arr)
    local inputs = {patch, epi}
    
    local targets = -self:get_disp(id, self.disp_arr) + 1 + self.disp_max -- not considering convolution
          
   return inputs, targets
end

function sup1Patch1EpiSet:sample(batchsize)
   
   self._indices = self._indices or torch.LongTensor()
   self._indices:resize(batchsize):random(1,self:size())
   return self:index(self._indices)

end

function sup1Patch1EpiSet:shuffle()
  
   local indices = torch.LongTensor():randperm(self:size())
   self.id = self.id:index(1,indices);
   return self, indices

end

function sup1Patch1EpiSet:split(ratio)
   
   assert(ratio > 0 and ratio < 1, "Expecting 0 < arg < 1")
   
   local size = self:size()
   local sizeA = math.floor(size*ratio)
   
   local loaders = {}
   for i,split in ipairs{{1,sizeA},{sizeA+1,size}} do
      local start, stop = unpack(split)
      local loader = dl.sup1Patch1EpiSet(self.img1_arr, self.img2_arr, self.disp_arr, self.hpatch)
      loader.id = self.id[{{start, stop}}]:clone();  
      loaders[i] = loader
   end
   return unpack(loaders)

end

function sup1Patch1EpiSet:size()
    return self.id:numel()
end

