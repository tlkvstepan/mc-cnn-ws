local trainerNet = {}

function trainerNet.getMilContrastive(disp_max, width, th_sup, th_occ, loss_margin, embed_net, head_net)
  
  local Net = nn.Sequential()
  local comNet = nn.ConcatTable()
  Net:add(comNet); 
  
  local refPosNet = nn.Sequential()
  local refNegNet = nn.Sequential()
  local negPosNet = nn.Sequential()
  
  comNet:add(refPosNet)
  comNet:add(refNegNet)
  comNet:add(negPosNet)
  
  local refPosSelNet = nn.ConcatTable()  -- input selectors for each distance net
  local refNegSelNet = nn.ConcatTable()
  local negPosSelNet = nn.ConcatTable()
 
  refPosNet:add(refPosSelNet)
  refNegNet:add(refNegSelNet)
  negPosNet:add(negPosSelNet)

  refPosSelNet:add(nn.SelectTable(1))
  refPosSelNet:add(nn.SelectTable(2))
  
  refNegSelNet:add(nn.SelectTable(1))
  refNegSelNet:add(nn.SelectTable(3))
  
  negPosSelNet:add(nn.SelectTable(3))
  negPosSelNet:add(nn.SelectTable(2))
    
  local refPosMetricNet = cnnMetric.setupSiamese(embed_net, head_net, width, disp_max)  
  local refNegMetricNet = cnnMetric.setupSiamese(embed_net, head_net, width, disp_max)  
  local negPosMetricNet = cnnMetric.setupSiamese(embed_net, head_net, width, disp_max)  
    
  refPosNet:add(refPosMetricNet);
  refNegNet:add(refNegMetricNet);
  negPosNet:add(negPosMetricNet);
  
  Net:add(nn.milContrastive(th_sup, th_occ, disp_max));
  
  local milFwdCst = nn.MarginRankingCriterion(loss_margin);
  local milBwdCst = nn.MarginRankingCriterion(loss_margin);
  local contrastiveFwdCst = nn.MarginRankingCriterion(loss_margin);
  local contrastiveBwdCst = nn.MarginRankingCriterion(loss_margin);
  local criterion = nn.ParallelCriterion():add(milFwdCst,1):add(milBwdCst,1):add(contrastiveFwdCst,1):add(contrastiveFwdCst,1)

  return Net, criterion

end

function trainerNet.getPipeline(disp_max, width, th_sup, loss_margin, embed_net, head_net)
    
  --[[
  Input: {{ref, pos}, matchInRow_pipe}, where ref, neg are tensor 1 x (2*hpatch + 1) x width 
  ]]--
  
  local Net = nn.Sequential()
  local comNet = nn.ParallelTable()
  Net:add( comNet )
  
  local metricNet = cnnMetric.setupSiamese(embed_net, head_net, width, disp_max) ;
  comNet:add( metricNet )  
  comNet:add( nn.Identity() )
  --Net:add( metricNet )
  Net:add( nn.pipeline(th_sup) ) 

  local fwdCst = nn.MarginRankingCriterion(loss_margin);
  local bwdCst = nn.MarginRankingCriterion(loss_margin);
  local criterion = nn.ParallelCriterion():add(fwdCst,1):add(bwdCst,1)

  return Net, criterion
end

function trainerNet.getMil(disp_max, width, loss_margin, embed_net, head_net)
    
  --[[
  Input: {ref, pos, neg}, where ref, neg, pos are tensor 1 x (2*hpatch + 1) x width 
  ]]--
  
  local Net = nn.Sequential()
  local comNet = nn.ConcatTable()
  Net:add(comNet); 
  
  local refPosNet = nn.Sequential()
  local refNegNet = nn.Sequential()
  local negPosNet = nn.Sequential()
  
  comNet:add(refPosNet)
  comNet:add(refNegNet)
  comNet:add(negPosNet)
  
  local refPosSelNet = nn.ConcatTable()  -- input selectors for each distance net
  local refNegSelNet = nn.ConcatTable()
  local negPosSelNet = nn.ConcatTable()
 
  refPosNet:add(refPosSelNet)
  refNegNet:add(refNegSelNet)
  negPosNet:add(negPosSelNet)

  refPosSelNet:add(nn.SelectTable(1))
  refPosSelNet:add(nn.SelectTable(2))
  
  refNegSelNet:add(nn.SelectTable(1))
  refNegSelNet:add(nn.SelectTable(3))
  
  negPosSelNet:add(nn.SelectTable(3))
  negPosSelNet:add(nn.SelectTable(2))
    
  local refPosMetricNet = cnnMetric.setupSiamese(embed_net, head_net, width, disp_max)  
  local refNegMetricNet = cnnMetric.setupSiamese(embed_net, head_net, width, disp_max)  
  local negPosMetricNet = cnnMetric.setupSiamese(embed_net, head_net, width, disp_max)  
    
  refPosNet:add(refPosMetricNet);
  refNegNet:add(refNegMetricNet);
  negPosNet:add(negPosMetricNet);
  
  Net:add(nn.mil(disp_max));
  
  local milFwdCst = nn.MarginRankingCriterion(loss_margin);
  local milBwdCst = nn.MarginRankingCriterion(loss_margin);
  local criterion = nn.ParallelCriterion():add(milFwdCst,1):add(milBwdCst,1)

  return Net, criterion
end  

function trainerNet.getContrastive(disp_max, width, th_sup, loss_margin, embed_net, head_net)
    
    
  local Net = cnnMetric.setupSiamese(embed_net, head_net, width, disp_max)    
  
  Net:add(nn.contrastive(th_sup, disp_max))  
    
  local contrastiveFwdCst = nn.MarginRankingCriterion(loss_margin);
  local contrastiveBwdCst = nn.MarginRankingCriterion(loss_margin);  

  local criterion = nn.ParallelCriterion():add(contrastiveFwdCst, 1):add(contrastiveBwdCst, 1)
  
  return Net, criterion
end

function trainerNet.getContrastiveDP(disp_max, width, th_sup, th_occ, loss_margin, embed_net, head_net)  

local Net = cnnMetric.setupSiamese(embed_net, head_net, width, disp_max)  

Net:add(nn.contrastiveDP(th_sup, th_occ))

local contrastiveFwdCst = nn.MarginRankingCriterion(loss_margin);
local contrastiveBwdCst = nn.MarginRankingCriterion(loss_margin);
local criterion = nn.ParallelCriterion():add(contrastiveFwdCst,1):add(contrastiveBwdCst,1)

return Net, criterion
end 

 
return trainerNet
