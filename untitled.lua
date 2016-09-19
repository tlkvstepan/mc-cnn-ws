require 'nn'
model = nn.Sequential()
model:add(nn.Linear(3,5))

prl = nn.ConcatTable()
prl:add(nn.Linear(5,1))
prl:add(nn.Linear(5,1))

model:add(prl)

criterion = nn.ParallelCriterion()
criterion:add(nn.MSECriterion())
criterion:add(nn.MSECriterion())

input = torch.rand(5,3)

target = {torch.rand(5,1),torch.rand(5,1)}

output = model:forward(input)
err = criterion:forward(output,target)

t = criterion:backward(output, target)
print(t)