require 'torch'

function fromfile(fname)
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




data_dir = 'data/MB'
te = fromfile(('%s/te.bin'):format(data_dir))
metadata = fromfile(('%s/meta.bin'):format(data_dir))
nnz_tr = fromfile(('%s/nnz_tr.bin'):format(data_dir))
nnz_te = fromfile(('%s/nnz_te.bin'):format(data_dir))
fname_submit = {}
for line in io.open(('%s/fname_submit.txt'):format(data_dir), 'r'):lines() do
  table.insert(fname_submit, line)
end
X = {}
dispnoc = {}
height = 1500
width = 1000
for n = 1,metadata:size(1) do
  local XX = {}
  light = 1
  while true do
    fname = ('%s/x_%d_%d.bin'):format(data_dir, n, light)
    if not paths.filep(fname) then
      break
    end
    table.insert(XX, fromfile(fname))
    light = light + 1
  end
  table.insert(X, XX)

  fname = ('%s/dispnoc%d.bin'):format(data_dir, n)
  if paths.filep(fname) then
    table.insert(dispnoc, fromfile(fname))
  end
end

dofile('CUnsupMB.lua')


mb = unsupMB(X, metadata, 5)
data = mb:get(128)
