require 'libdprog'
require 'gnuplot'

local E = torch.rand(30,50):float()
local aE =  torch.zeros(30,50):float()
local aP = torch.zeros(30,50):float()
local aS = torch.zeros(30,50):float()
local T = torch.zeros(30,50):float()

indices = torch.Tensor(30):float()
values = torch.Tensor(30):float()

dprog.compute(E, aE, aS, aP, T);
gnuplot.imagesc(E,'color')
gnuplot.figure()
gnuplot.imagesc(T,'color')
print(T)
