require 'torch'
require 'gnuplot'
require 'autograd'

params = {}

function isValid(x, y, w, h, disp_max)

  if( x >= y ) or 
    ( x < (y - disp_max) ) or
    ( x < 1 ) or
    ( y < 1 ) or
    ( x > w ) or
    ( y > h ) then 
      return false
    else 
      return true
    end
    
end

function simulate(w, h, disp_max)
  
  local E = torch.Tensor(h,w):fill(1)
  local dx = torch.Tensor{1,0,1}
  local dy = torch.Tensor{0,1,1}
    
  local x = 1
  local y = torch.random(x + 1, x + disp_max) 
  E[y][x] = 0;
  
  repeat
    repeat
      idx = torch.random(1,3)
      yy = y + dy[idx]
      xx = x + dx[idx]
    until isValid(xx, yy, w, h, disp_max)
    E[yy][xx] = 0
    x = xx;
    y = yy;
  until (x == w) or (y == h)
  E = E + 0.5*torch.rand(h, w)
  
  return E
end

function track(aE, aL, T, disp_max)
  
  local h, w = E:size(1), E:size(2)
  local dx = torch.Tensor{-1, 0,-1}
  local dy = torch.Tensor{ 0,-1,-1}
  local x, y
  local P = E:clone():zero()
  y = h
  
  local E_vit = 1/0
  local x0 = 1/0
  for x = (w - disp_max), w-1 do 
      local E = aE[y][x] / aL[y][x]
      if( E_vit > E ) then
        E_vit = E
        x0 = x
      end
  end
    
  y = h  
  x = x0  
  P[y][x] = 1
  n = T[y][x]
  while n > -1 do
    x = x + dx[n]
    y = y + dy[n]
    P[y][x] = 1
    n = T[y][x]
  end
  
  return E_vit, P
end  
  
function accum(E, disp_max)

  local h, w = E:size(1), E:size(2)
  local dx = torch.Tensor{-1, 0,-1}
  local dy = torch.Tensor{ 0,-1,-1}
  local x, y, xx, yy
  local aE = E:clone():zero() 
  local aL = E:clone():zero()
  local T = E:clone():zero()
  
  -- init first col
  x = 1
  for y = 2, x + disp_max do
    if isValid(x, y, w, h, disp_max) then
      aE[y][x] = E[y][x]
      aL[y][x] = 1
      T[y][x] = -1
    end
  end
    
  for x = 2, w-1 do
    local y_min = x + 1
    local y_max = x + disp_max
    for y = y_min, y_max do
      if isValid(x, y, w, h, disp_max) then 
        local n_best = 1/0
        local avgE_best = 1/0
        local aE_best= 1/0
        local aL_best= 1/0
        for n = 1, 3 do
          local xx = x + dx[n]
          local yy = y + dy[n]
          if isValid(xx, yy, w, h, disp_max) then
            avgE = (aE[yy][xx] + E[y][x]) / (aL[yy][xx] + 1)
            if avgE < avgE_best then
              n_best = n
              avgE_best = avgE
              aE_best= aE[yy][xx] + E[y][x]
              aL_best= aL[yy][xx] + 1
            end
          end
        end
        aE[y][x] = aE_best
        aL[y][x] = aL_best
        T[y][x] = n_best 
      end
    end
  end
  
  return aE, aL, T
end

function viterbi(param, disp_max)
  local aE, aL, T = accum(E, disp_max)
  local E_vit, P = track(aE, aL, T, disp_max)
  return E_vit
end

--param = {
--  E =  
--}

w, h, disp_max = 100, 100, 30
E = simulate(w, h, disp_max)
gnuplot.figure(1)
gnuplot.imagesc(E,'color')
aE, aL, T = accum(E, disp_max)
E_vit, P = track(aE, aL, T, disp_max)
gnuplot.figure(2)
gnuplot.imagesc(aE,'color')



