-- code fix
1. test Mil, Contrastive and ContrastiveMil
2. add Middlebury and KITTI15 datasets
3. tune stereo pipeline with best of our nets (1000 random parameters samples)
4. make equi-performance curve on our test set (for best of our nets and supervised)    

-- experiments    
0. check training time for mc-cnn
1. train our best performing net on KITTI15
2. train our best performing net on Middlebury
3. compare cross-performance KITTI12-KITTI15-MB
4. tune stereo pipeline with our best net for KITTI12 KITTI15 and Middlebury
5. compare stereo performance on KITTI12, KITTI15 and Middlebury
6. make equperformance curve for our best net and supervised net
7. make mil-max-contrast-dprog network

-- minor code improvements

-- ideas
1. learn smoothness cost and reiterate (self learning)
2. training data is errorneous
3. impose order: closer to true maximim higher similarity
