-- libraries:
t = require 'torch'
grad = require 'autograd'


  local aW = torch.Tensor(3)    

-- define model
neuralNet = function(params, x)
  aW[1] = params.W[1];
  return torch.Tensor{1}
end

-- gradients:
dneuralNet = grad(neuralNet, {optimize=true})

-- define trainable parameters:
params = {
   W = t.randn(5),
}

-- compute loss and gradients wrt all parameters in params:
dparams, loss = dneuralNet(params, 3)
print(dparams.W)
-- in this case:
--> loss: is a scalar (Lua number)
--> dparams: is a table that mimics the structure of params; for
--  each Tensor in params, dparams provides the derivatives of the
--  loss wrt to that Tensor.