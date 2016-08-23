--[[
  
  Given number of stereo images **CLASS** provides access to dataset that consists of
  1 reference patch and 2 epipolar stripes: 
  
  _refPatch_ is nb_stripes x (2*haptch + 1) x (2*haptch + 1) tensor
  _posEpi_, _negEpi_ are nb_stripes x (2*hpatch+1) x 2*hpatch + disp_max tensors,
  
  _posEpi_ is epipolar stripe where we can find match for _refPatch_
  _negEpi_ is epipolar stripe where we can't find match for _refPatch_
  
  This class will sit on CPU. 
  
]]--
  
  local dl = require 'dataload'
  local unsup1Patch2EpiSet = torch.class('dl.unsup1Patch2EpiSet', 'dl.DataLoader', dl)

  function unsup1Patch2EpiSet:__init( img1_arr, img2_arr, hpatch, disp_max)
    
    -- img1_arr, img2_arr - N x img_h x img_w tensors of stereo images
    -- hpatch - half patch size ((2*hpatch+1) is size of patch) 
    -- disp_max - maximum disparity value (disparity is shift to the left on img2) 
    
    self.img1_arr = img1_arr;
    self.img2_arr = img2_arr;

    self.nb_pairs = img1_arr:size(1);
    self.img_h = img1_arr:size(2);
    self.img_w = img1_arr:size(3);
    self.hpatch = hpatch; 
    self.disp_max = disp_max;
    
    self.id = self:get_valid_id(); -- make list of all ids of avaliable epipolar stripes
    
  end

  
function unsup1Patch2EpiSet:pair_col_row_2_id(pair, col, row) 
  
  -- Function convert image pair number, x and y of patch center to patch id
  -- Patches are indexed in row-first order
  
  col = col - 1;
  row = row - 1;
  pair = pair - 1;
  local id = (pair * self.img_h + row) * self.img_w + col
  local id = id + 1;
  
  return id

end

function unsup1Patch2EpiSet:id_2_pair_col_row(id)
  
  -- Function convert patch's id to pair number, x and y
  -- Patches are indexed in row-first order
  
  id = id - 1;
  local area = (self.img_w * self.img_h)
  local pair = torch.floor(id / area);
  local reminder = (id % area);
  local y = torch.floor(reminder / self.img_w);
  local x = reminder % self.img_w;
  
  return pair + 1, x + 1, y + 1;

end
  
function unsup1Patch2EpiSet:get_valid_id()
  
  --[[ Function make list of valid patch indices 
       Patch is valid if its (x,y) of its center 
       is x in [hpatch + disp_max + 1, img_w - hpatch] and 
          y in [hpatch + 1, img_h - hpatch]
  ]]--
  
     -- firstly make valid indices for single image
     local x = torch.range(self.hpatch + self.disp_max + 1, self.img_w - self.hpatch):int();
     x = x:view(1,x:numel()):clone();
     local y = torch.range(self.hpatch + 1, self.img_h - self.hpatch):int();
     y = y:view(y:numel(),1):clone();
     local xx = torch.repeatTensor(x,y:size(1),1);
     local yy = torch.repeatTensor(y,1,x:size(2));
     xx = xx:view(xx:numel(),1);
     yy = yy:view(yy:numel(),1)
          
     -- compute indices for all images
     local id = torch.IntTensor(xx:numel()*self.nb_pairs);
     for npair = 1,self.nb_pairs do
          id[{{(npair-1)*xx:numel()+1, npair*xx:numel()}}] = self:pair_col_row_2_id(npair, xx, yy);
     end
     return id
     
end

function unsup1Patch2EpiSet:get_epi(id, img)
    
    -- given id(s) returns epipolar stripes
        
    local pair, col, row =  self:id_2_pair_col_row(id);
    local row_max = row + self.hpatch; 
    local row_min = row - self.hpatch;
    
    local epi = torch.Tensor(id:numel(), 2*self.hpatch+1, self.disp_max + 2*self.hpatch + 1)
    for n = 1,id:numel() do
      local col_max = col[n] + self.hpatch;
      local col_min = col[n] - self.disp_max - self.hpatch;
      epi[{{n},{},{}}] = torch.squeeze(img[{{pair[n]},{row_min[n],row_max[n]},{col_min, col_max}}]);
    end
          
    return epi;
  
end

function unsup1Patch2EpiSet:get_patch(id, img)
  
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

function unsup1Patch2EpiSet:index(indices, inputs, targets)   
    
    --[[ 
    Given array of indices return _inputs_ and _targets_
    _inputs_ is table of 3 tensors: _refEpi(s)_, _posEpi(s)_, _negEpi(s)_  
       
       _posEpi_, _negEpi_ are nb_stripes x (2*hpatch+1) x max_disp + 2*hpatch tensors,
       _refPatch_ are nb_stripes x (2*hpatch+1) x (2*hpatch+1) tensors
        
    _targets_ table of table of nb_stripes x 1 ones
    ]]--  

    local nb_indices = indices:numel()
    local id_ref = {}
    local id = {}

    --[[ 
    As negative example we select epipolar stripe that corresponds to another valid
    patch from the set. This epipolar stripe is shifted by at least |half of patch|
    from the positive epipolar.
    ]]--

    for n = 1, nb_indices do
      id_ref[n] = self.id[indices[n]]
      repeat
        id[n] = self.id[math.random(1, self.id:numel())]
        local pair, col, row = self:id_2_pair_col_row(id[n])
        local pair_ref, col_ref, row_ref = self:id_2_pair_col_row(id_ref[n])
        local row_diff = math.abs(row-row_ref);
      until ((row_diff > self.hpatch) or (pair_ref ~= pair) ) 
    end  

    local ref_patch = self:get_patch(torch.Tensor(id_ref), self.img1_arr)
    local pos_epi = self:get_epi(torch.Tensor(id_ref), self.img2_arr)
    local neg_epi = self:get_epi(torch.Tensor(id), self.img2_arr)
    
    inputs = {ref_patch, pos_epi, neg_epi}   
    targets = torch.ones(1,nb_indices)
    
    return inputs, targets;

  end

  function unsup1Patch2EpiSet:sample(batchsize)

    self._indices = self._indices or torch.LongTensor()
    self._indices:resize(batchsize):random(1, self:size())
    
    return self:index(self._indices)

  end

  function unsup1Patch2EpiSet:shuffle()
    
    local indices = torch.LongTensor():randperm(self:size())
    self.id = self.id:index(1,indices);
    
    return self, indices
  
  end

  function unsup1Patch2EpiSet:split(ratio)

    assert(ratio > 0 and ratio < 1, "Expecting 0 < arg < 1")

    local size = self:size()
    local sizeA = math.floor(size*ratio)

    local loaders = {}
    for i,split in ipairs{{1,sizeA},{sizeA+1,size}} do
      local start, stop = unpack(split)
      local loader = dl.unsup1Patch2EpiSet(self.img1_arr, self.img2_arr, self.hpatch, self.disp_max)
      loader.id = self.id[{{start, stop}}]:clone();  
      loaders[i] = loader
    end
    return unpack(loaders)

  end

  function unsup1Patch2EpiSet:size()
    return self.id:numel()
  end






