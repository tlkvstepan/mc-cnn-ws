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

function utils.read_log(filename, sep)
  local file = io.open(filename)
  table = {}
  if file then
      nline = 0
      for line in file:lines() do
        nline = nline + 1; 
        table[nline] = {}
        nsub_line = 0
        for sub_line in string.gmatch(inputstr, "([^"..sep.."]+)") do
            nsub_line = nsub_line + 1
            table[nline][nsub_line] = sub_line
        end
      end
  end
end

function utils.save_tensor(filename, tensor)
 
  tensor:add(-tensor:min())
  tensor:div(tensor:max()-tensor:min())
  image.save(filename, torch.squeeze(tensor));

end

return utils