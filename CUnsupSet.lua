--[[
  
  This is prototype for unsupervised training set 
  
  Methods:
  
  **init( )**  
    
    Initialize unsupervised set. Should be redifined.  
  
  **subset( prc )**
  
    Command to use only prc*100 % of set.
  
  **get( batchSize )** 
  
    Given _batchSize_ returns _batchInput_, _width_, _dispMax_ 
    
    _batchInput_ is table that consists of _posEpi_, _refEpi_, _negEpi_
    _posEpi_, _negEpi_ are _batchSize_ x (2*hpatch+1) x width tensors,
    _refEpi_ are _batchSize_ x (2*hpatch+1) x (width - max_disp) tensors. 
    _posEpi_ is conjugate epipolar line for _refEpi_ epipolar line.
    _negEpi_ is not conjugate epipolar line for _refEpi_.
 
    _width_ epipolar line length
    _dispMax_ maximum disparity
    
]]--

  local unsupSet = torch.class('unsupSet')

  function unsupSet:__init()
    
  end
  
  function unsupSet:get( batch_size )
    
  end

  function unsupSet:subset( prc )
    
  end



