--[[
  
  Given number of stereo images **CLASS** provides access to dataset that 
  consists of two epipolar line and gt disparity:     
  
  _refEpi_ and _posEpi_ are nb_patch x (2*hpatch+1) x (disp_max + 2*hpatch + 1) tensors,
    
  _gtDisp_ is nb_patch x disp_max tensor with ground truth disparities (or inf if disparity is not known)
   
  Motivation: compute test results faster, visualize "true" match line 
]]--

local sup2EpiSet = torch.class('sup2EpiSet', 'DataLoader')
local image = require 'image'

function sup2EpiSet:__init(img1_arr, img2_arr, disp_arr, hpatch)
   
  disp_arr = torch.round(disp_arr);
  
  self.img1_arr = img1_arr;
  self.img2_arr = img2_arr;
  self.disp_arr = disp_arr; 
   
  self.nb_pairs = img1_arr:size(1);
  self.img_h = img1_arr:size(2);
  self.img_w = img1_arr:size(3);
  self.hpatch = hpatch; 
  self.disp_max = disp_arr:max();
    
  -- get list of valid epipolar lines
  self.id = self:get_valid_id();
  
end


function sup2EpiSet:id_2_pair_row(id)
    
  -- given id(s) function returns pair(s)# and row(s)#
    
  local pair = torch.floor( (id - 1) / self.img_h) + 1;
  local row   = (id - 1) % self.img_h + 1;
  return pair, row;
    
end

function sup2EpiSet:pair_row_2_id(pair, row)
    
  -- given pair(s)# and row(s)# function returns id(s)
    
  local id = row + self.img_h*(pair-1) ;
  return id;
  
end


function sup2EpiSet:get_gt(id, disp)

  -- given id(s) returns true disparities
  
  local pair, row =  self:id_2_pair_row(id);
  local col = torch.range(1, self.img_w)
  local mask_computable = col:gt(self.hpatch):cmul(col:le(self.img_w - self.hpatch))
  local mask_bound = col:gt(self.hpatch):cmul(col:le(self.img_w - self.hpatch))
  local gt = torch.Tensor(id:numel(), 1, self.img_w-2*self.hpatch)
  
  for n = 1, id:numel() do
      -- we keep data in float, but when queried convert to double 
      local epi_disp = disp[{{pair[n]},{row[n]},{}}]:squeeze():double()
      local mask_computable_ = mask_computable:clone():cmul((epi_disp+self.hpatch):lt(col))
      local mask_uncomputable = -mask_computable_ + 1
      local mask_unknown = epi_disp:le(0.5)
      local mask = mask_uncomputable + mask_unknown
      mask = mask:gt(0)
      epi_disp[mask] = -1;
      epi_disp = epi_disp[mask_bound] 
      gt[{{n},{}}] = epi_disp
  end
  
  return gt
end  
 
function sup2EpiSet:get_epi(id, img)
    
    -- given id(s) returns epipolar stripes
  
    pair, row =  self:id_2_pair_row(id);
    local row_max = row + self.hpatch; 
    local row_min = row - self.hpatch;
    
    local epi = torch.Tensor(id:numel(), 2*self.hpatch+1, self.img_w)
    for n = 1, id:numel() do
      -- we keep data in float, but when queried convert to double 
      epi[{{n},{},{}}] = img[{{pair[n]},{row_min[n],row_max[n]},{}}]:double();
    end
    
    return epi;
  
end

function sup2EpiSet:get_valid_id()
  
  --[[ 
    
    Function make list of valid epipolar line indices 
    Epipolar line is valid if:
    1) it has at least 1 known disparity;
    2) it's y is in [hpatch+1, height - hpatch]. 
   
  ]]--
  
    local x = torch.range(1, self.img_w);
    x = x:view(1,x:numel()):clone();
    local y = torch.range(1, self.img_h);
    y = y:view(y:numel(),1):clone();
    local xx = torch.repeatTensor(x,y:size(1),1);
    local yy = torch.repeatTensor(y,1,x:size(2));
    
    local mask_computable = xx:gt(self.hpatch)
    mask_computable = mask_computable:cmul(xx:le(self.img_w - self.hpatch))
    mask_computable = mask_computable:cmul(yy:le(self.img_h - self.hpatch))
    mask_computable = mask_computable:cmul(yy:gt(self.hpatch))
    
    local id = {}
    for npair = 1,self.nb_pairs do
      
      local disp = torch.squeeze(self.disp_arr[{{npair},{},{}}]):double();
      local mask_computable_ = mask_computable:clone():cmul((disp+self.hpatch):lt(xx))
      local mask_unknown = disp:le(0.5)
      
      local mask_known = -mask_unknown + 1;
      
      local mask = mask_known:cmul(mask_computable_)
      local mask = mask:max(2):gt(0)
      local active_row = torch.range(1,self.img_h)[mask]:clone()
      
      id[npair] = self:pair_row_2_id(npair, active_row);
    
    end
    
    id = torch.cat(id, 1);
   
    return id
end

function sup2EpiSet:index(indices, inputs, targets)   
   
   --[[ 
    
    Given array of indices return _inputs_ and _targets_
    
    _inputs_ is table of 2 tensors: _refEpi(s)_, _posEpi(s)_  
    _posEpi_, _negEpi_ are nb_examples x (2*hpatch+1) x img_w tensors,
        
    _targets_ is nb_examples x img_w tensor of disparities 
    Invalid disparities have inf value..
    
    Note that disparities are valid if they are 
    1) known 
    2) computable, i.e. x > disp_max
    
    ]]--  

    local nb_indices = indices:numel()
    local id = self.id:index(1, indices:long())
    
    local epiRef = self:get_epi(id, self.img1_arr)
    local epiPos = self:get_epi(id, self.img2_arr)
    local inputs = {epiRef, epiPos}
    
    local targets = self:get_gt(id, self.disp_arr)   
          
   return inputs, targets
end


function sup2EpiSet:sample(batchsize)
   
   self._indices = self._indices or torch.LongTensor()
   self._indices:resize(batchsize):random(1,self:size())
   return self:index(self._indices)

end

function sup2EpiSet:shuffle()
  
   local indices = torch.LongTensor():randperm(self:size())
   self.id = self.id:index(1,indices);
   return self, indices

end

function sup2EpiSet:split(ratio)
   
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

function sup2EpiSet:size()
    return self.id:numel()
end

