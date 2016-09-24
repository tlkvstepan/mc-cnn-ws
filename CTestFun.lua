local testFun = {}

--[[

  This is function for testing network accuracy. In takes as input _testNet_, _input_, _target_
  
  _testNet_ is test network
  _input_ is table consisting of _patch_ and _epi_ tensors
    _patch_ is nb_samples x (2*hpatch + 1) x (2*hpatch + 1) tensor 
    _epi_ is nb_samples x (2*hpatch + 1) x (2*hpatch + 1 + disp_max) tensor 
  _target_ is gt disparities of _patch_ in _epi_.

  You should take care about parameters of network!
  
]]--

function testFun.patchEpiEval(testNet, input, target)

  local nb_samples = input[1]:size(1);
  local patch, epi = unpack(input) 
  local hpatch = (patch:size(2)-1)/2
  
  local dispErr  = {};
  
  norm2 = nn.Normalize(2);
  unsqueeze = nn.Unsqueeze(1);
  
  if testNet:type() == "torch.CudaTensor" then
    unsqueeze:cuda()
    norm2:cuda()
  end
  
  local ref = {}
  local sol = {}
  local gt = {}
  local dispDiff = {}
  local nb_ge5 = 0
  for nsample = 1, nb_samples do

    local gtDisp = target[nsample];
    local patchDisc = testNet:forward(patch[{{nsample},{},{}}]):clone()
    local epiDisc   = testNet:forward(epi[{{nsample},{},{}}]):clone()
    
    patchDisc = patchDisc:squeeze()
    epiDisc = epiDisc:squeeze()
    
    patchDisc = patchDisc / patchDisc:norm()
    epiDisc = norm2:forward(epiDisc:t()):t()
    
    local cosSim = unsqueeze:forward(patchDisc)*epiDisc;
    local val, disp = cosSim:max(2)
        
    dispErr[nsample] = math.abs(disp:squeeze() - gtDisp) 
    
    if dispErr[nsample] >= 3 then
        nb_ge5 = nb_ge5 + 1;
        ref[nb_ge5] = patch[{{nsample},{},{}}]:double();
        sol[nb_ge5] = epi[{{nsample},{},{disp:squeeze(), disp:squeeze()+2*hpatch}}]:double();
        gt[nb_ge5] = epi[{{nsample},{},{gtDisp, gtDisp+2*hpatch}}]:double();
        dispDiff[nb_ge5] = dispErr[nsample];
    end
      
  end
  
  errCases = {torch.cat(ref,1), torch.cat(sol,1), torch.cat(gt,1), torch.Tensor(dispDiff)}
  dispErr = torch.Tensor(dispErr)
  acc_lt3 = dispErr[dispErr:lt(3)]:numel() * 100 / dispErr:numel();
  acc_lt5 = dispErr[dispErr:lt(5)]:numel() * 100 / dispErr:numel();
  
  return acc_lt3, acc_lt5, errCases  
end


--[[

  This is function for testing network accuracy. In takes as input _testNet_, _input_, _target_
  
  _distNet_ is network that computes distance matrix for two epipolar lines
  
  _input_ is table consisting of _refEpi_ and _posEpi_ tensors
    _refEpi_ is nb_samples x (2*hpatch + 1) x (im_w) tensor 
    _posEpi_ is nb_samples x (2*hpatch + 1) x (im_w) tensor 
  
  _target_ is gt disparities of _patch_ in _epi_ (with inf in unknow val).

  You should take care about parameters of network!
  
]]--

function testFun.epiEpiEval(distNet, input, target)

  local nb_samples = input[1]:size(1);
  local refEpi, posEpi = unpack(input) 
  local hpatch = (refEpi:size(2)-1)/2
  local im_w = refEpi:size(3)
  local row = torch.range(1, im_w - 2*hpatch);
  
--  local dispErr  = {};
  
--  norm2 = nn.Normalize(2);
--  unsqueeze = nn.Unsqueeze(1);
  
--  if testNet:type() == "torch.CudaTensor" then
--    unsqueeze:cuda()
--    norm2:cuda()
--  end
  
--  local ref = {}
--  local sol = {}
--  local gt = {}
  local dispErr = {}
--  local nb_ge5 = 0
  
  for nsample = 1, nb_samples do

    local gtDisp      = target[{{nsample}, {}}]:squeeze();
    
    local sample_input = {refEpi[{{nsample}, {}, {}}], posEpi[{{nsample}, {}, {}}]}
    local distMat = distNet:forward(sample_input) 
    
    local val, wta_sol = torch.max(distMat, 2)
    wta_sol = wta_sol:double()
    wta_sol = torch.add(row,-wta_sol) -- here we compute real disparities 
    
    local sampleDispDiff = torch.abs(wta_sol-gtDisp)
    sampleDispDiff = sampleDispDiff[gtDisp:ne(-1)]
    
    dispErr[nsample] = sampleDispDiff;
    
--    local refEpiDisc  = testNet:forward(patch[{{nsample}, {}, {}}]):clone()
--    local posEpiDisc  = testNet:forward(epi[{{nsample}, {}, {}}]):clone()
    
--    patchDisc = patchDisc:squeeze()
--    epiDisc = epiDisc:squeeze()
    
--    patchDisc = patchDisc / patchDisc:norm()
--    epiDisc = norm2:forward(epiDisc:t()):t()
    
--    local cosSim = unsqueeze:forward(patchDisc)*epiDisc;
--    local val, disp = cosSim:max(2)
        
--    dispErr[nsample] = math.abs(disp:squeeze() - gtDisp) 
    
--    if dispErr[nsample] >= 3 then
--        nb_ge5 = nb_ge5 + 1;
--        ref[nb_ge5] = patch[{{nsample},{},{}}]:double();
--        sol[nb_ge5] = epi[{{nsample},{},{disp:squeeze(), disp:squeeze()+2*hpatch}}]:double();
--        gt[nb_ge5] = epi[{{nsample},{},{gtDisp, gtDisp+2*hpatch}}]:double();
--        dispDiff[nb_ge5] = dispErr[nsample];
--    end
      
  end
  
  dispErr  = torch.cat(dispErr,1)
  
--  errCases = {torch.cat(ref,1), torch.cat(sol,1), torch.cat(gt,1), torch.Tensor(dispDiff)}
--  dispErr = torch.Tensor(dispErr)
  acc_lt3 = dispErr[dispErr:lt(3)]:numel() * 100 / dispErr:numel();
  acc_lt5 = dispErr[dispErr:lt(5)]:numel() * 100 / dispErr:numel();
  
--  return acc_lt3, acc_lt5, errCases  
end

return testFun