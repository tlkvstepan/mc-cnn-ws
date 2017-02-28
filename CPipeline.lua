
--[[
  
  Given similarity w x w matrix and pipeline solution w - 2 x hpatch as input, module outputs table with two tensors: E_maxInRow, E_maxInCol and E_pipe
    
    E_greedy is h tensor, that consists of row-wise greedy matching cost 
    E_maxInRow, E_maxInCol is h tensor, that consists of row-wise cost of pipeline solution

  Before computing greedy solution we all elements of the similartiy matrixmask within th_sup radius from the pipeline solution

--]]

local pipeline, parent = torch.class('nn.pipeline', 'nn.Module')

function pipeline:__init(th_sup)
   parent.__init(self)
   
   self.th_sup    = th_sup
   self.disp_max  = disp_max
   
   self.matchRow_pipe = torch.Tensor()
   self.matchCol_pipe   = torch.Tensor()
   self.I_maxInRow_greedy = torch.Tensor()
   self.I_maxInCol_greedy = torch.Tensor()
end

function pipeline:updateOutput(input)
  
  local simMatFwd, matchCol_pipe = unpack(input)
  self.matchCol_pipe = matchCol_pipe:cuda()
  local simMatBwd = simMatFwd:clone() 
  
  local dim = simMatFwd:size(1)
  
  local valid = self.matchCol_pipe:lt(1/0)
    
  self.matchRow_pipe = torch.range(1, dim):cuda()    
  self.matchRow_pipe = self.matchRow_pipe[valid]
  self.matchCol_pipe = self.matchCol_pipe[valid] 
  
  if self.matchRow_pipe:numel() > 1 then
  
    local simMatFwd_vec = simMatFwd:view(dim*dim)
    local simMatBwd_vec = simMatBwd:view(dim*dim)
    
    local E_pipe = simMatFwd_vec:index(1, (self.matchCol_pipe + (self.matchRow_pipe-1)*dim):cudaLong()) 
      
    for d = -self.th_sup, self.th_sup do
      
       -- fwd
       local ind =  self.matchCol_pipe + d
       ind[ind:lt(1)] = 1
       ind[ind:gt(dim)] = dim
       simMatFwd_vec:indexFill(1, (ind + ( self.matchRow_pipe - 1 )*dim):cudaLong(), -1/0 )
       
       -- bwd 
       ind = self.matchRow_pipe + d
       ind[ind:lt(1)] = 1
       ind[ind:gt(dim)] = dim
       simMatBwd_vec:indexFill(1, (self.matchCol_pipe + ( ind - 1 )*dim):cudaLong(), -1/0 )
       
    end
    
    local E_maxInRow_greedy, I_maxInRow_greedy = torch.max(simMatFwd, 2)
    local E_maxInCol_greedy, I_maxInCol_greedy = torch.max(simMatBwd, 1)
    
    E_maxInRow_greedy = E_maxInRow_greedy:squeeze()
    I_maxInRow_greedy = I_maxInRow_greedy:squeeze():cuda()
    E_maxInCol_greedy = E_maxInCol_greedy:squeeze()
    I_maxInCol_greedy = I_maxInCol_greedy:squeeze():cuda()
            
    
    
    -- remain only el that exist in the pipeline solution
    E_maxInRow_greedy = E_maxInRow_greedy:index(1,self.matchRow_pipe:cudaLong()) 
    self.I_maxInRow_greedy = I_maxInRow_greedy:index(1,self.matchRow_pipe:cudaLong())
    --
    E_maxInCol_greedy = E_maxInCol_greedy:index(1,self.matchCol_pipe:cudaLong()) 
    self.I_maxInCol_greedy = I_maxInCol_greedy:index(1,self.matchCol_pipe:cudaLong()) 
    --
    self.output = {{E_pipe, E_maxInRow_greedy}, {E_pipe, E_maxInCol_greedy}}
  
  else
      
    self.output = {}
    
  end
  
  return self.output
end

function pipeline:updateGradInput(input, gradOutput)
  
  local simMat, matchCol_pipe = unpack(input)
  local pipeFwd, pipeBwd = unpack(gradOutput)
  
  local ograd_pipe1, ograd_maxInRow = unpack(pipeFwd)
  local ograd_pipe2, ograd_maxInCol = unpack(pipeBwd)
   
  -- pass input gradient to max and second max 
  --self.gradInput:resizeAs(simMat):zero()
  local dim = simMat:size(1)
     
  local igrad_simMat = simMat:clone():zero()  
  local igrad_simMat_vec = igrad_simMat:view(dim*dim) 
  local idx
  
  idx = self.matchCol_pipe + (self.matchRow_pipe-1)*dim;
  igrad_simMat_vec:indexAdd(1, idx, ograd_pipe1:squeeze() + ograd_pipe2:squeeze())
  
  -- fwd
  idx = self.I_maxInRow_greedy + (self.matchRow_pipe-1)*dim;
  igrad_simMat_vec:indexAdd(1, idx, ograd_maxInRow:squeeze())
  
  -- bwd
  idx = self.matchCol_pipe + (self.I_maxInCol_greedy-1)*dim;
  igrad_simMat_vec:indexAdd(1, idx, ograd_maxInCol:squeeze())
    
  self.gradInput = {igrad_simMat, matchCol_pipe:zero()}  
  
  return self.gradInput 
end

