-- code fix
0. use all test set for performance evaluation
1. add Middlebury and KITTI15 datasets
2. tune stereo pipeline with best of our nets (1000 random parameters samples)
3. make equi-performance curve on our test set (for best of our nets and supervised)    

-- experiments    
0. check training time for mc-cnn
1. train contrast-dprog, mil-dprog and mil-contrast-dprog nets.
2. test all our nets and supervised net on our test set
3. train our best performing net on Middlebury
4. tune stereo pipeline with our best net for KITTI12 KITTI15 and Middlebury
5. compare stereo performance on KITTI12, KITTI15 and Middlebury
6. make equperformance curve for our best net and supervised net

-- minor code improvements
1. save net together with optim state and train parameters (it will allow to simply restart learning). 
2. make two axis plot for optim logger (will allow to show to scale for train and test error)
3. save timestamp in optim logger (it will help to do early stopping).

-- ideas
1. learn smoothness cost and reiterate (self learning)
2. training data is errorneous
3. impose order: closer to true maximim higher similarity
4. mil-dprog might be not optimal due to large number of occlusion in ref-neg ( use hybrid? )