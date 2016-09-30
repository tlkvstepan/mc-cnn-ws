-- code fix
1. make mil-dprog function
2. make mil-contrast-dprog function 
3. fix test net function
4. add Middlebury and KITTI15 datasets
6. tune stereo pipeline with best of our nets (1000 random parameters samples)
7. compare best of our net with supervised
8. make equi-performance curve on our test set (for best of our nets and supervised)    

-- experiments    
1. train mil-contrast-max, contrast-dprog, mil-dprog and mil-contrast-dprog nets.
2. perform optimal stopping for every net
3. test all our nets and supervised net on our test set
4. train our best performing net on Middlebury
5. tune stereo pipeline with our best net for KITTI12 KITTI15 and Middlebury
6. compare stereo performance on KITTI12, KITTI15 and Middlebury
7. make equperformance curve for our best net and supervised net

-- ideas
1. learn smoothness cost and reiterate (self learning)
2. training data is errorneous
3. impose order: closer to true maximim higher similarity