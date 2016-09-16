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

function testFun.epiEval(testNet, input, target)

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

return testFun