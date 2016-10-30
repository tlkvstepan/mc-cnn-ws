require 'torch'
require 'gnuplot'

x = torch.randperm(100)
x = x[{{1,100}}]
vals = torch.Tensor(100000)
for i = 1,100000 do
  vals[i] = x[torch.random(1,100)];
end

H = torch.histc(vals)
gnuplot.plot(H)