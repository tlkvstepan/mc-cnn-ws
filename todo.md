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
4. Tune stereo paremeters for our descriptor (1000 random samples)
5. Compare against mc-cnn on stereo
6. Implement cost that maximize difference between first and second maximums
7. Implement entropy cost  
8. Test how both costs behave on their own.
9. Try to combine them with MIL objective.
   
   
% ideas
1. dynamic programming (self learning)
2. learn smoothness cost and reiterate (self learning)
3. training data is errorneous


  
