require 'torch'

dofile('CPBIL.lua')


prm = {hight=torch.rand(3), width=torch.rand(2)}
cost = torch.rand(10);

pbil_opt = pbil(prm)
sample=pbil_opt:sample()
pbil_opt:update(cost)




