require 'torch'
require 'nn'
require 'cunn'
require 'gnuplot'
dofile('copyElements.lua')
dofile('fixedIndex.lua')
nnMetric = dofile('nnMetric.lua')

width = 500;
dispMax = 3;
hpatch = 5

------------------------------------------------------------
------------------ compute masks
------------------------------------------------------------

left  = torch.rand(1, hpatch*2+1, width ) 
right = torch.rand(1, hpatch*2+1, width ) 

row = torch.Tensor(width-hpatch*2,1)
row[{{},{1}}] = torch.range(1, width-hpatch*2)
rows = torch.repeatTensor(row, 1, width-hpatch*2)

col = row:t():clone()
cols = torch.repeatTensor(col, width-hpatch*2, 1)

disp = rows - cols 
mask = disp:le(dispMax):cmul( disp:gt(0) )

indices = (cols-1) + (rows-1)*(width-hpatch*2) + 1
indices = indices[mask] --:view((width-hpatch*2)*(width-hpatch*2),1)


activeCols = cols[mask]
activeRows = rows[mask]
nbPairs = mask:ne(0):sum()

-------------------------------------------------------------
------------- embedding and feature nets
------------------------------------------------------------

local nbConvLayers = 5  
local nbFeatureMap = 64
local kernel = 3
embNet0 = nnMetric.mccnnEmbeddNet( nbConvLayers, nbFeatureMap, kernel )

-----------------------------------------------------------
------------------ pairs selector
-----------------------------------------------------------

local net =  nn.Sequential()
local selector =  nn.ParallelTable()
net:add(selector)

embNet0:add(nn.Squeeze(2))
embNet0:add(nn.Transpose({1,2}))
embNet1 = embNet0:clone('weight','bias', 'gradWeight','gradBias');
embNet0:add(nn.fixedIndex(1, activeRows:long()))
embNet1:add(nn.fixedIndex(1, activeCols:long()))

selector:add(embNet0)
selector:add(embNet1)

-------------------------------------------------------------
------------------ similarity computation
-------------------------------------------------------------

   
-------------------------------------------------------------
------------------- cosine 
-------------------------------------------------------------

simNet = nn.Sequential()
normNet = nn.ParallelTable()
simNet:add(normNet)
normNet:add(nn.Normalize(2))
normNet:add(nn.Normalize(2))
simNet:add(nn.DotProduct())
net:add(simNet)

-------------------------------------------------------------
------------------- linear 
-------------------------------------------------------------


--local simNet =  nn.Sequential()
--simNet:add( nn.JoinTable(2) )
--nbFcLayers = 4
--nbFcUnits = 384
--simNet:add( nn.ReLU() )
--for nFcLayer = 1,nbFcLayers do
--   local idim = (nFcLayer == 1) and nbFeatureMap*2 or nbFcUnits
--   local odim = nbFcUnits
--   simNet:add( nn.Linear(idim, odim) )
--   simNet:add( nn.ReLU(true) )
--end
--simNet:add( nn.Linear(nbFcUnits, 1) )
--simNet:add( nn.Sigmoid(true) )
--simNet:add( nn.Squeeze(2) )
--net:add(simNet)

----------------------------------------------------------------
-------------------- make similarity matrix
----------------------------------------------------------------


net:add(nn.copyElements(torch.LongStorage{nbPairs}, torch.LongStorage{width-hpatch*2, width-hpatch*2}, torch.range(1, nbPairs), indices))





net:cuda()
out = net:forward({left:cuda(),right:cuda()})

gradOut = torch.rand(net.output:size())
out = net:backward({left:cuda(),right:cuda()}, gradOut)


gnuplot.imagesc(out:double(), gray)
x = 'end'

