require 'libdprog'
require 'gnuplot'

local E = torch.rand(5,5):float()
local path = torch.zeros(5,5):float()
costRef2pos = torch.Tensor(5):float()
costPos2ref = torch.Tensor(5):float()
indexRef2pos = torch.Tensor(5):float()
indexPos2ref = torch.Tensor(5):float()

dprog.compute(E, path, costRef2pos, costPos2ref, indexRef2pos, indexPos2ref);
gnuplot.imagesc(E,'color')
gnuplot.figure()
gnuplot.imagesc(path,'color')
print(path)
