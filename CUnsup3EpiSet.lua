--[[
  
  Given number of stereo images **CLASS** provides access to dataset that 
  consists of 3 epipolar stripes:     
  
  _posEpi_, _negEpi_ are nb_stripes x (2*hpatch+1) x img_w tensors,
  _refEpi_ are nb_stripes x (2*hpatch+1) x (img_w - max_disp) tensors. 
  
  _posEpi_ is conjugate epipolar line for _refEpi_ epipolar line.
  _negEpi_ is not conjugate epipolar line for _refEpi_.
 
]]--

  local dl = require 'dataload'
  local unsup3EpiSet = torch.class('dl.unsup3EpiSet', 'dl.DataLoader', dl)

  function unsup3EpiSet:__init( img1_arr, img2_arr, hpatch, disp_max)
    
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

  function unsup3EpiSet:id_2_pair_row(id)
    
    -- given id(s) function returns pair(s)# and row(s)#
    
    local pair = torch.floor( (id - 1) / self.img_h) + 1;
    local row   = (id - 1) % self.img_h + 1;
    return pair, row;
    
  end

  function unsup3EpiSet:pair_row_2_id(pair, row)
    
    -- given pair(s)# and row(s)# function returns id(s)
    
    local id = (pair-1)*self.img_h + (row - 1) + 1;
    return id;
  
  end
  
   
  function unsup3EpiSet:get_epi(id, img)
    
    -- given id(s) returns epipolar stripes
  
    pair, row =  self:id_2_pair_row(id);
    local row_max = row + self.hpatch; 
    local row_min = row - self.hpatch;
    
    local epi = torch.Tensor(id:numel(), 2*self.hpatch+1, self.img_w)
    for n = 1,id:numel() do
      epi[{{n},{},{}}] = img[{{pair[n]},{row_min[n],row_max[n]},{}}];
    end
    return epi;
  
  end


  function unsup3EpiSet:get_valid_id()
    
    -- makes list of valid ids of epipolar stripes
    -- (all pixels of valid epipolar stripe are valid)
    
    -- firstly make valid row indices for single image
    local row = torch.range(self.hpatch + 1, self.img_h - self.hpatch);

    -- add indexes for all images
    local id = torch.Tensor(row:numel()*self.nb_pairs);
    for npair = 1,self.nb_pairs do
      id[{{(npair-1)*row:numel()+1, npair*row:numel()}}] = self:pair_row_2_id(npair, row);
    end
    return id
  
  end


  function unsup3EpiSet:index(indices, inputs, targets)   
    --[[ 
    Given array of indices return _inputs_ and _targets_
    _inputs_ is table of 3 tensors: _posStripe(s)_, _negStripe(s)_, _refStripe(s)_  
       
       _posStripes_, _negStripes_ are nb_stripes x (2*hpatch+1) x img_w tensors,
       _refStripes_ are nb_stripes x (2*hpatch+1) x (img_w - max_disp) tensors
        
    _targets_ table of table of nb_stripes x (img_w-max_disp) ones
    ]]--  

    local nb_indices = indices:numel()
    local id_ref = {}
    local id = {}

    --[[ 
    As negative example select epipolar stripe that corresponds to anothe valid
    id from the set. Check that this epipolar stripe is shifted by at least |half of patch|
    from the positive example.
    ]]--

    for n = 1, nb_indices do
      id_ref[n] = self.id[indices[n]]
      repeat
        id[n] = self.id[math.random(1, self.id:numel())]
        local id_diff = math.abs(id_ref[n]-id[n]);
      until (id_diff > self.hpatch) 
    end  

    local ref_epi = self:get_epi(torch.Tensor(id_ref), self.img1_arr)
    local pos_epi = self:get_epi(torch.Tensor(id_ref), self.img2_arr)
    local neg_epi = self:get_epi(torch.Tensor(id), self.img2_arr)
    inputs = {ref_epi[{{},{},{self.disp_max+1,ref_epi:size(3)}}], pos_epi, neg_epi}   
    targets = torch.ones(nb_indices, self.img_w - self.disp_max - 2*self.hpatch)
    
    return inputs, targets;

  end

  function unsup3EpiSet:sample(batchsize)

    self._indices = self._indices or torch.LongTensor()
    self._indices:resize(batchsize):random(1, self:size())
    
    return self:index(self._indices)

  end

  function unsup3EpiSet:shuffle()
    
    local indices = torch.LongTensor():randperm(self:size())
    self.id = self.id:index(1,indices);
    
    return self, indices
  
  end

  function unsup3EpiSet:split(ratio)

    assert(ratio > 0 and ratio < 1, "Expecting 0 < arg < 1")

    local size = self:size()
    local sizeA = math.floor(size*ratio)

    local loaders = {}
    for i,split in ipairs{{1,sizeA},{sizeA+1,size}} do
      local start, stop = unpack(split)
      local loader = dl.unsup3EpiSet(self.img1_arr, self.img2_arr, self.hpatch, self.disp_max)
      loader.id = self.id[{{start, stop}}]:clone();  
      loaders[i] = loader
    end
    return unpack(loaders)

  end

  function unsup3EpiSet:size()
    return self.id:numel()
  end






