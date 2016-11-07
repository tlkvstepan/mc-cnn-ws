--[[ 
This is universal script for semi-supervised training of network 

It supports following networks:

-mil-max
-mil-dprog
-contrast-max
-contrast-dprog
-mil-contrast-max
-mil-contrast-dprog

]]--

--  -debug_start_from_fnet  work/contrast-dprog-mb/fnet_2016_11_02_19:05:06_contrast-dprog-mb.t7 -debug_start_from_optim work/contrast-dprog-mb/optim_2016_11_02_19:05:06_contrast-dprog-mb.t7



---debug_start_from_fnet  work/test-mb-load/fnet_2016_11_04_14:40:02_test-mb-load.t7 -debug_start_from_optim work/test-mb-load/optim_2016_11_04_14:40:02_test-mb-load.t7
require 'torch'

-- |read input parameters|
-- fist   - argument is debug mode
-- second - training wrapper architecture
-- third  - training set
dbg = table.remove(arg, 1) 
if dbg == 'debug' then
  dbg = true
else
  dbg = false
end

cmd = torch.CmdLine()

-- optimization parameters parameters
if not dbg then
  cmd:option('-valid_set_size', 100)        -- 100 epi lines      
  cmd:option('-train_batch_size', 369)      -- 342 one image in KITTI
  cmd:option('-train_nb_batch', 90)        -- 400 all images in KITTI
  cmd:option('-train_nb_epoch', 100)        -- 35 times all images in KITTI
else
  cmd:option('-valid_set_size', 100)       -- 100 epi lines      
  cmd:option('-train_batch_size', 128)     -- 129 one image in KITTI
  cmd:option('-train_nb_batch', 1)       -- 50
  cmd:option('-train_nb_epoch', 10)        -- 10
end
cmd:option('-train_set_prc', 0.1)        -- 100 epi lines      
 
-- training network parameters
cmd:option('-loss_margin', 0.2)
cmd:option('-maxsup_th', 2) -- 2
cmd:option('-occ_th', 1)    -- 1

-- feature network parameters
-- (we have different default networks for mb and kitti)
cmd:option('-net_nb_feature', 64)
cmd:option('-net_kernel', 3)
cmd:option('-net_nb_layers', 5)

-- debug
cmd:option('-debug_fname', 'test-mb')
cmd:option('-debug_start_from_fnet', '')
cmd:option('-debug_start_from_optim', '')

prm = cmd:parse(arg)

paths.mkdir('work/'..prm['debug_fname']); -- make output folder
print('Semi-suprevised training contrst-dp arhitecture and mb  set\n')
  
-- |load modules|

-- standard modules
require 'gnuplot'
require 'optim'
require 'nn'
require 'image'
require 'lfs' -- change current lua directory

-- Custom modules
dofile('CAddMatrix.lua')                  -- Module that adds constant matrix to the input (I use it for masking purposes)

require 'libdprog'                        -- C++ module for dynamic programming
dofile('CContrastDprog.lua');             -- Contrastive dynamic programming module
dofile('CContrastMax.lua');               -- Contrastive max-2ndMax module
dofile('CMilDprog.lua');
dofile('CMilContrastDprog.lua')
dofile('DataLoader.lua');                 -- Parent class for dataloaders

-- unsupervised set difinition
dofile('CUnsupSet.lua')     
dofile('CUnsupMB.lua')    -- MB

dofile('CUnsupKITTI_HD.lua')

--dofile('CUnsup3EpiSet.lua');              -- Unsupervised training set loader
dofile('CSup2EpiSet.lua');          -- Supervised validation set loader

baseNet = dofile('CBaseNet.lua');         -- Function that makes base net
netWrapper = dofile('CNetWrapper.lua')    -- Function that "wrap" base net into training net
testFun = dofile('CTestUtils.lua');         -- Function that performs test on validation set

utils = dofile('utils.lua');              -- Utils for loading and visualization

print('Parameters of the procedure : \n')
utils.printTable(prm)
        
require 'cunn'
require 'cudnn'

math.randomseed(0); 
torch.manualSeed(0)

-- |define feature network|
-- If we choose to start from timestamp, when try to read pre-trained base feature net
local fnet_fname = prm['debug_start_from_fnet'] 
local optim_fname = prm['debug_start_from_optim'] 
_BASE_FNET_, hpatch = baseNet.get(prm['net_nb_layers'], prm['net_nb_feature'], prm['net_kernel'])
_OPTIM_STATE_ = {}
if utils.file_exists(fnet_fname) and utils.file_exists(optim_fname)   then
  print('Continue training.\n')
  _BASE_FNET_= torch.load(fnet_fname, 'ascii')
  hpatch = ( utils.get_window_size(_BASE_FNET_)-1 )/ 2
  _OPTIM_STATE_ = torch.load(optim_fname, 'ascii')
else
  print('Start training from the begining\n')
end

-- put base net and state of optimizer on gpu if needed
if _OPTIM_STATE_.m then
    _OPTIM_STATE_.m = _OPTIM_STATE_.m:cuda()
    _OPTIM_STATE_.v = _OPTIM_STATE_.v:cuda()
    _OPTIM_STATE_.denom = _OPTIM_STATE_.denom:cuda()
end

-- get training and base network parametesr
_BASE_FNET_:cuda()
_BASE_PPARAM_, _BASE_PGRAD_ = _BASE_FNET_:getParameters() 

-- |read dataset|
do
  
  local metadata_fname = 'data/mb/meta.bin'
  local metadata = utils.fromfile(metadata_fname)
  local img_tab = {}
  local disp_tab = {}
  for n = 1,metadata:size(1) do
    local img_light_tab = {}
    light = 1
    while true do
      fname = ('data/mb/x_%d_%d.bin'):format(n, light)
      if not paths.filep(fname) then
        break
      end
      table.insert(img_light_tab, utils.fromfile(fname))
      light = light + 1
    end
    table.insert(img_tab, img_light_tab)
    fname = ('data/mb/dispnoc%d.bin'):format(n)
    if paths.filep(fname) then
      table.insert(disp_tab, utils.fromfile(fname))
    end
    if metadata[{n,3}] == -1 then -- fill max_disp for train set
      metadata[{n,3}] = disp_tab[n]:max()
    end
  end
    
  unsupSet = unsupMB(img_tab, metadata, hpatch)

end

-- |define optimization function|
feval = function(x)

  local batch_size = #_TR_INPUT_[1];
  local epiRef, epiPos, epiNeg = unpack(_TR_INPUT_) -- epiNeg does not exist for  contrast-max and contrast-dprog
  
  _TR_ERR_ = 0;   
  
  -- clear gradients
  _BASE_PGRAD_:zero()
  
  local nb_comp_ttl = 0; 
  local criterion, tr_net
  
  for nsample = 1, batch_size do
  
    local sample_input = {}
    sample_input[1] = epiRef[nsample]
    sample_input[2] = epiPos[nsample]
    if epiNeg ~= nil then
      sample_input[3] = epiNeg[nsample]
    end
    
    if( _WIDTH_TAB_[nsample] ~= _WIDTH_TAB_[nsample-1] or _DISP_TAB_[nsample] ~= _DISP_TAB_[nsample-1] )  then
    
    -- make training network (note that parameters and gradients are copied from base network
      tr_net, criterion =  netWrapper.getContrastDprog(_WIDTH_TAB_[nsample], _DISP_TAB_[nsample], hpatch, prm['maxsup_th'],  prm['occ_th'],  prm['loss_margin'], _BASE_FNET_:clone():double())
      
      -- put network on cuda
      tr_net:cuda()
      criterion:cuda()
      
      tr_param, tr_grad = tr_net:getParameters() 
      
    else
      
      tr_grad:zero()
    
    end
    
    -- forward pass through net
    tr_net:forward(sample_input)
    
    -- find number of nonempty output tabels
    local nb_tables = #tr_net.output
    
    -- if nuber of nonempty output tables is 0, we can not do anything
    if nb_tables  > 0 then
    if tr_net.output[1][1]:numel() > 1  then
     
      -- make target array for every table, and simultaneously compute 
      -- total number of samples
      local sample_target = {};
      for ntable = 1,nb_tables do
        local nb_comp = tr_net.output[ntable][1]:numel()
        nb_comp_ttl = nb_comp_ttl + nb_comp;
        sample_target[ntable] = nn.utils.addSingletonDimension(tr_net.output[ntable][1]:clone():fill(1),1) 
        sample_target[ntable] = sample_target[ntable]:cuda()
      end
      
      -- pass through criterion
      _TR_ERR_ = _TR_ERR_ + criterion:forward(tr_net.output, sample_target)

      -- backword pass
      tr_net:backward(sample_input, criterion:backward(tr_net.output, sample_target))
       collectgarbage()
    
    end
    end
  
    -- copy gradients to base net
    _BASE_PGRAD_:add(tr_grad)
    
  end
 
  _TR_ERR_ = _TR_ERR_ / nb_comp_ttl
  _BASE_PGRAD_:div(nb_comp_ttl);

  return _TR_ERR_, _BASE_PGRAD_      
end

-- |save debug info|
local timestamp = os.date("%Y_%m_%d_%X_")
torch.save('work/' .. prm['debug_fname'] .. '/params_' .. timestamp .. prm['debug_fname'] .. '.t7', prm, 'ascii');
    
-- |define logger|
logger = optim.Logger('work/' .. prm['debug_fname'] .. '/'.. prm['debug_fname'], true)
logger:setNames{'Training loss', 'Accuracy (<3 disparity err)'}
logger:style{'+-', '+-'}

-- |optimize network|   
local start_time = os.time()

-- go through epoches
for nepoch = 1, prm['train_nb_epoch'] do

  nsample = 0;
  sample_err = {}
  local train_err = 0
  
  -- go through batches
  for nbatch = 1, prm['train_nb_batch'] do
    
   -- get batch
   _TR_INPUT_, _WIDTH_TAB_, _DISP_TAB_ = unsupSet:get(prm['train_batch_size'])   
           
   -- some network need only two epopolar lines
   if arch == 'contrast-max' or arch == 'contrast-dprog' then
      _TR_INPUT_[3] = nil
   end
      
   -- put on gpu
   for nsample = 1, prm.train_batch_size do
    for i = 1,#_TR_INPUT_ do
      _TR_INPUT_[i][nsample] = _TR_INPUT_[i][nsample]:cuda()
    end
   end
       
   -- optimize 
   optim.adam(feval, _BASE_PPARAM_, {}, _OPTIM_STATE_)    
   table.insert(sample_err, _TR_ERR_)
   
   collectgarbage()
   
  end


train_err = torch.Tensor(sample_err):mean();

local end_time = os.time()
local time_diff = os.difftime(end_time,start_time);

local timestamp = os.date("%Y_%m_%d_%X_")
  
-- save net (we save all as double tensor)
local net_fname = 'work/' .. prm['debug_fname'] .. '/fnet_' .. timestamp .. prm['debug_fname'] .. '.t7';
torch.save(net_fname, _BASE_FNET_:clone():double(), 'ascii');

-- save optim state (we save all as double ten_TE_TARGET_sor)
local optim_state_host ={}
optim_state_host.t = _OPTIM_STATE_.t;
optim_state_host.m = _OPTIM_STATE_.m:clone():double();
optim_state_host.v = _OPTIM_STATE_.v:clone():double();
optim_state_host.denom = _OPTIM_STATE_.denom:clone():double();
torch.save('work/' .. prm['debug_fname'] .. '/optim_'.. timestamp .. prm['debug_fname'] .. '.t7', optim_state_host, 'ascii');

-- compute test error using MC-CNN
local str = './main.lua mb our -a test_te -sm_terminate cnn -net_fname ../mil-mc-cnn/' .. net_fname 

-- run for validation subset and get :error
lfs.chdir('../mc-cnn')
local handle = io.popen(str)
local result = handle:read("*a")
str_err = string.gsub(result,'\n','');
test_err = tonumber(str_err);
lfs.chdir('../mil-mc-cnn')

-- save log
logger:add{train_err, test_err}
logger:plot()

-- print
print(string.format("epoch %d, time = %f, train_err = %f, test_acc = %f", nepoch, time_diff, train_err, test_err))
--rint(string.format("epoch %d, time = %f, train_err = %f", nepoch, time_diff, train_err))

collectgarbage()

end





