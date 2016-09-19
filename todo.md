*1. make use of second matrix
*2. plot distance from the correct positions for errors
*3. large scale learning (all KITTI)
*4. visualize distance matrix: mask it, compute softmax
*5. set up processing on cluster

--- 07.09
*1. Correct logging when continue training 
*2. During trainig save all history of debug info (net, distance, error cases)
   (this will allow to make early stopping)
*3. Retrain net
*4. Implement cost that maximize difference between first and second maximums
*5. Implement net that efficiently combine two cost computation
*6. Implement alternative testing and get results for learned net (several error thresholds)

7. Implement and test contrastive net with dynamic programing only
8. Implement contrastive net with mil




8. Equi-performance curve   
9. Tune stereo paremeters for our descriptor (1000 random samples)
10. Compare against mc-cnn on stereo
11. Test how both costs behave on their own.
12. Try to combine them with MIL objective.
13. Use all training data for KITTI
14. Make validation and test set to tune parameters (such as early stopping)
15. Set up experiment for Middlebury, Stretcha datasets
16. Modify optim logger
17. Use cmd parameter logger
% ideas
0. oversharpening of maximum. since ground truth is not precise some positives might be cosidered as negatives
   maybe our algorithm does better than ground truth?
   we need to change test procedure.. focus on number of % of correct disparities (withing threashold).
   maybe we need to tune maxinum order and margin
1. dynamic programming (self learning)
2. learn smoothness cost and reiterate (self learning)
3. training data is errorneous
5. impose order: closer to true maximim higher similarity

  
