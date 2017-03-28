require 'libdprog_kbest'  -- C++ DP
require 'gnuplot'
require 'image'
utils = dofile('utils.lua')

dispMax = 228
pathNum = 20
dim = 100

E = 1-image.load('test_dp.png',1,'byte'):float():squeeze()/255
dim = E:size(1)
--gnuplot.imagesc(E, 'color')

--E = torch.FloatTensor(dim, dim);
aE = torch.FloatTensor(pathNum, dim, dim);
T  = torch.FloatTensor(pathNum, dim, dim);
P  = torch.FloatTensor(pathNum, dim, dim);

--dprog_kbest.simulate(E, dispMax)
dprog_kbest.compute(E, P, aE, T, dispMax);


allP = P:sum(1);
gnuplot.imagesc(allP:squeeze(), 'color')

for i = 1,pathNum do
  fname = 'path' .. i .. '.png'
  image.save(fname, P[i])
end
