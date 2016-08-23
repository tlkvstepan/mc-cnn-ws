--[[
  
  Given number of stereo images **CLASS** provides access to dataset that 
  consists of 3 patches:     
  
  _refPatch_, _posPatch_, _negPatch_ are nb_patch x (2*hpatch+1) x (2*hpatch+1) tensors,
  
  _posPatch_ is conjugate patch for _refPatch_.
  _negPatch_ is not conjugate patch for _refEpi_.
 
]]--
local dl = require 'dataload'
local sup3PatchSet = torch.class('dl.sup3PatchSet', 'dl.DataLoader', dl)

function sup3PatchSet:__init(img1_arr, img2_arr, disp_arr, hpatch)
   
  disp_arr = torch.round(disp_arr);
  
  self.img1_arr = img1_arr;
  self.img2_arr = img2_arr;
  self.disp_arr = disp_arr; 
   
  self.nb_pairs = img1_arr:size(1);
  self.img_h = img1_arr:size(2);
  self.img_w = img1_arr:size(3);
  self.hpatch = hpatch; 
  self.disp_max = disp_arr:max();
    
  -- initialize list of valid ids
  self.id = self:get_valid_id();
  
end


function sup3PatchSet:pair_col_row_2_id(pair, col, row) 
  
  -- Function convert image pair number, x and y of patch center to patch id
  -- Patches are indexed in row-first order
  col = col - 1;
  row = row - 1;
  pair = pair - 1;
  local id = ( (pair * self.img_h + row) * self.img_w + col )
  local id = id + 1;
  
  return id

end

function sup3PatchSet:id_2_pair_col_row(id)
  
  -- Function convert patch's id to pair number, x and y
  -- Patches are indexed in row-first order
  
  id = id - 1;
  
  local area = (self.img_w * self.img_h)
  local pair = ( torch.floor(id / area) );
  local reminder = (id % area);
  local y =  torch.floor(reminder / self.img_w);
  local x = reminder % self.img_w ;
  
  return (pair + 1), (x + 1), (y + 1);

end
  

function sup3PatchSet:get_patch(id, img)
  
  -- Function returns patches when given id 
  
  local pair, col, row =  self:id_2_pair_col_row(id);
  local row_max = row + self.hpatch; 
  local row_min = row - self.hpatch;
  local col_max = col + self.hpatch; 
  local col_min = col - self.hpatch;
  
  local patch = torch.Tensor(id:numel(), 2*self.hpatch+1, 2*self.hpatch+1)
    
  for n = 1,id:numel() do
      patch[{{n},{},{}}] = torch.squeeze(img[{{pair[n]},{row_min[n],row_max[n]},{col_min[n], col_max[n]}}]);
  end
  
  return patch
end

function sup3PatchSet:get_valid_id()
  --[[ Function make list of valid patch indices 
       Patch is valid if its (x,y) of its center 
       is x in [hpatch + 1 + 1 (we need at least one negative!), img_w - hpatch] and  
          y in [hpatch + 1, img_h - hpatch]
       and its disparity is known       
  ]]--
  
    -- make list of indices for 1st image
    local x = torch.range(1, self.img_w);
    x = x:view(1,x:numel()):clone();
    local y = torch.range(1, self.img_h);
    y = y:view(y:numel(),1):clone();
    local xx = torch.repeatTensor(x,y:size(1),1);
    local yy = torch.repeatTensor(y,1,x:size(2));
         
    -- add indexes for all images
    local id = {}
    for npair = 1, self.nb_pairs do
          
      -- get disparity map
      local disp = torch.squeeze(self.disp_arr[{{npair},{},{}}]);
          
      -- find poinrs where disp is not zero
      local mask = torch.squeeze(disp:ge(1));
      if mask then 
          
          -- find x and y of these points 
            local cur_xx = xx[mask];
            local cur_yy = yy[mask];
            local cur_disp = disp[mask];
            
            -- find points where we can fit patch in img1
            local mask = cur_xx:gt(self.hpatch + 1) -- to ensure that there is at least one negative 
            local mask_ = cur_xx:le(self.img_w-self.hpatch)
            mask:cmul(mask_);
            local mask_ = cur_yy:gt(self.hpatch)
            mask:cmul(mask_);
            local mask_ = cur_yy:le(self.img_h-self.hpatch)
            mask:cmul(mask_);
            local mask_ = torch.add(cur_xx,-cur_disp):gt(self.hpatch)
            mask:cmul(mask_);
                        
            -- find x, y for these points
            local cur_xx = cur_xx[mask];
            local cur_yy = cur_yy[mask];
                        
            -- find ID
            local cur_ID = self:pair_col_row_2_id(npair, cur_xx, cur_yy);
            
            table.insert(id, cur_ID) 
      end
    end
    
    local id = torch.cat(id, 1);
    return id
end

function sup3PatchSet:index(indices, inputs, targets)   
   
   --[[ 
    Given array of indices return _inputs_ and _targets_
    _inputs_ is table of 3 tensors: _refPatch(s)_, _posPatch(s)_, _negPatch(s)_  
       
       _posEpi_, _negEpi_ are nb_patch x (2*hpatch+1) x (2*hpatch+1) tensors,
        
    _targets_ table of table of nb_patch x 1 ones
    ]]--  

    local nb_indices = indices:numel()
    local id_ref = {}
    local id_neg = {}
    local id_pos = {}
    
    --[[ 
    As negative example we select from the same epipolar line |half of patch|
    ]]--
    
    for n = 1, nb_indices do
      id_ref[n] = self.id[indices[n]]
      local pair_ref, col_ref, row_ref = self:id_2_pair_col_row(id_ref[n])
      local gt_disp = torch.squeeze(self.disp_arr[{{pair_ref},{row_ref},{col_ref}}]);
      local disp
      repeat
        local cur_disp_max = (col_ref - self.hpatch - 1) < self.disp_max and col_ref - self.hpatch - 1 or self.disp_max
        disp = math.random(0, cur_disp_max )
      until ( disp ~= gt_disp ) 
      id_neg[n] = self:pair_col_row_2_id(pair_ref, col_ref - disp, row_ref); 
      id_pos[n] = self:pair_col_row_2_id(pair_ref, col_ref - gt_disp, row_ref); 
    end  
    
    id_pos = torch.Tensor(id_pos)
    id_neg = torch.Tensor(id_neg)
    id_ref = torch.Tensor(id_ref)
    
    local ref_patch = self:get_patch(id_ref, self.img1_arr)
    local pos_patch = self:get_patch(id_pos, self.img2_arr)
    local neg_patch = self:get_patch(id_neg, self.img2_arr)
    
    inputs = {ref_patch, pos_patch, neg_patch}   
    targets = torch.ones(1,nb_indices)
      
   return inputs, targets
end

function sup3PatchSet:sample(batchsize)
   
   self._indices = self._indices or torch.LongTensor()
   self._indices:resize(batchsize):random(1,self:size())
   return self:index(self._indices)

end

function sup3PatchSet:shuffle()
  
   local indices = torch.LongTensor():randperm(self:size())
   self.id = self.id:index(1,indices);
   return self, indices

end

function sup3PatchSet:split(ratio)
   
   assert(ratio > 0 and ratio < 1, "Expecting 0 < arg < 1")
   
   local size = self:size()
   local sizeA = math.floor(size*ratio)
   
   local loaders = {}
   for i,split in ipairs{{1,sizeA},{sizeA+1,size}} do
      local start, stop = unpack(split)
      local loader = dl.sup3PatchSet(self.img1_arr, self.img2_arr, self.disp_arr, self.hpatch)
      loader.id = self.id[{{start, stop}}]:clone();  
      loaders[i] = loader
   end
   return unpack(loaders)

end

function sup3PatchSet:size()
    return self.id:numel()
end

