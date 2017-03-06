local utils = {}

require 'torch'
require 'image'

function utils.get_random_file_name()
    
    local timestamp = os.date("%Y_%m_%d_%X") 
    local random_key = math.random(1,100000000)
    
    return timestamp .. '_' .. random_key 
end

function utils.fromfile(fname)
  
   local file = io.open(fname .. '.dim')
   local dim = {}
   for line in file:lines() do
      table.insert(dim, tonumber(line))
   end
   if #dim == 1 and dim[1] == 0 then
      return torch.Tensor()
   end

   local file = io.open(fname .. '.type')
   local type = file:read('*all')

   local x
   if type == 'float32' then
      x = torch.FloatTensor(torch.FloatStorage(fname))
   elseif type == 'int32' then
      x = torch.IntTensor(torch.IntStorage(fname))
   elseif type == 'int64' then
      x = torch.LongTensor(torch.LongStorage(fname))
   else
      print(fname, type)
      assert(false)
   end

   x = x:reshape(torch.LongStorage(dim))
   return x
   
end

function utils.mask(im, disp_max)
  
  im_new = im:clone():double()
  img_w = im_new:size(2)
  local mask = torch.ones(img_w, img_w)  
  mask = torch.triu(torch.tril(mask,-1),-disp_max)
  m = im_new:mean()
  im_new:cmul(mask);
  mask:mul(m)
  mask:add(-m)
  im_new:add(-mask)
  
return im_new
end

function utils.softmax(im)
  im_new = im:clone():double()
  cols_max = im_new:max(2)
  im_new:add(-torch.repeatTensor(cols_max,1,im_new:size(2)))
  im_new:exp()
  cols_sum = im_new:sum(2)
  im_new:cdiv(torch.repeatTensor(cols_sum,1,im_new:size(2)))
  return im_new
end
 

function utils.scale2_01(im)
  
  local im_scale = im:clone()
  im_scale:add(-im_scale:min()):div(im_scale:max()-im_scale:min())
  
return im_scale

end
  
function utils.vis_errors(p1, p2, p3, text)
  
  -- ref, pos, neg are tensors nb_patches x h x w that we want to visualize 
  -- txt it table text we put on each patch row
 
  local h = p1:size(2)
  local w = p1:size(3)
  local nb_patch = p1:size(1)
  
  local border = 1
  local max_nb_patch = 50
   
  -- reshuffle all errorneous patches
  local idx = torch.LongTensor():randperm(nb_patch)
  if( nb_patch > max_nb_patch ) then
    nb_patch = max_nb_patch
  end
  
  local im = torch.Tensor(3, 3*(w+border)+border, (h+border)*nb_patch);
  idx = idx[{{1,nb_patch}}];
  
  for nsample = 1,nb_patch do
      
      local cur_idx = idx[nsample] 
      local cur_txt = tostring(text[idx[nsample]])
      local patch1 = p1[{{cur_idx},{},{}}]:double()
      local patch2 = p2[{{cur_idx},{},{}}]:double()
      local patch3 = p3[{{cur_idx},{},{}}]:double()
      
      patch1:add(-patch1:min())
      patch1:div(patch1:max()-patch1:min())
      patch2:add(-patch2:min())
      patch2:div(patch2:max()-patch2:min())
      patch3:add(-patch3:min())
      patch3:div(patch3:max()-patch3:min())
      
     -- num_txt = image.drawText(cur_txt, 2, 2, {color={0,0,0},size = 1})
      local line = torch.cat({torch.zeros(1,border,w), patch1, torch.zeros(1,border,w),
                   patch2, torch.zeros(1,border,w), patch3,torch.zeros(1,border,w)}, 2)
      line = torch.cat({torch.zeros(1,3*(h+border)+border,border), line}, 3)
      line = torch.repeatTensor(line,3,1,1)
      
      
      im[{{},{},{(nsample-1)*(w+border)+1,(nsample)*(w+border)}}] = line;          
      
  end
    
  return im;
end

function utils.printTable(par)
  -- function prints table 
  for key, val in pairs(par) do
      if type(val) == 'number' then
        str = string.format("%10s : %03.3f", key, val) 
      elseif type(val) == 'boolean' then
        str =  string.format("%10s : %s", key, tostring(val)) 
      elseif type(val) == 'string' then
        str =  string.format("%10s : %s", key, val) 
      end
      print(str)     
  end
end

function utils.file_exists(fname)
   local f=io.open(fname,"r")
   if f~=nil then io.close(f) return true else return false end
end

function utils.meshgrid(w,h)
  
  local x = torch.range(1, w);
  x = x:view(1,x:numel()):clone();
  local y = torch.range(1, h);
  y = y:view(y:numel(),1):clone();
  local xx = torch.repeatTensor(x,y:size(1),1);
  local yy = torch.repeatTensor(y,1,x:size(2));
  
  return xx, yy
end

function utils.get_window_size(net)
   local ws = 1
   for i = 1,#net.modules do
      local module = net:get(i)
      if torch.typename(module) == 'cudnn.SpatialConvolution' or torch.typename(module) == 'nn.SpatialConvolution' then
         ws = ws + module.kW - 1
      end
   end
   return ws
end

-- copy module parameters of net
function utils.copynet(dst, src)
  for i = 1,#src.modules do
    if torch.typename(src.modules[i]) == 'cudnn.SpatialConvolution' or torch.typename(src.modules[i]) == 'nn.SpatialConvolution' then
      dst.modules[i].weight:copy(src.modules[i].weight:double())
      dst.modules[i].bias:copy(src.modules[i].bias:double())
    end
  end
  return dst
end      

return utils