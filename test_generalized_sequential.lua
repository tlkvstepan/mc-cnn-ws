require 'nn'
dofile('CGeneralizedSequential.lua');

criterion = nn.GeneralizedSequential()
criterion:add(nn.SplitTable(1))
criterion:add(nn.MapTable():add(nn.SoftMax()))
criterion:add(nn.MapTable():add(nn.Replicate(2)))

input = 2*torch.rand(2,2)-1;
output = criterion:forward(input);

print(output)