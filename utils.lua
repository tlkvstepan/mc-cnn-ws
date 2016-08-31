local utils = {}

require 'torch'
require 'image'

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

--function utils.n_tensor(filename, tensor)
 
--  tensor:add(-tensor:min())
--  tensor:div(tensor:max()-tensor:min())
--  image.save(filename, torch.squeeze(tensor));

--end

return utils