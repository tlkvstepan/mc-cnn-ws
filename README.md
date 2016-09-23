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
*7. Implement and test contrastive net with dynamic programing only
*8. Implement contrastive net with mil
*9. Implement mil dprog

-- 21.09
*1. make test net function
1. fix mil-dprog
2. improve dprog function (evaluate once)
3. test set from 2 epipolar lines (use all points with known gt) 
4. draw actual epipolar line profile (or points)
5. prepare for experiments with middlebury dataset
6. tune stereo pipeline with best of our nets (1000 random parameters samples)
7. compare best of our net with supervised
8. make equi-performance curve on our test set (for best of our nets and supervised)    
    

% ideas
0. oversharpening of maximum. since ground truth is not precise some positives might be cosidered as negatives
   maybe our algorithm does better than ground truth?
   we need to change test procedure.. focus on number of % of correct disparities (withing threashold).
   maybe we need to tune maxinum order and margin
1. dynamic programming (self learning)
2. learn smoothness cost and reiterate (self learning)
3. training data is errorneous
4. impose order: closer to true maximim higher similarity