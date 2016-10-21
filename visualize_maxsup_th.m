close all;

% 1-pix occ
figure;
title('Occlusion threshold selection')
occ0File = 'work/maxsup-threshold-selection/contrast-max-0/contrast-max-0-2016_10_14_10:58:36'
occ1File = 'work/maxsup-threshold-selection/contrast-max-1/contrast-max-1-2016_10_14_11:15:20'
occ2File = 'work/maxsup-threshold-selection/contrast-max-2/contrast-max-2-2016_10_14_11:32:10'
occ4File = 'work/maxsup-threshold-selection/contrast-max-4/contrast-max-4-2016_10_14_11:49:13'
occ8File = 'work/maxsup-threshold-selection/contrast-max-8/contrast-max-8-2016_10_14_12:06:47'
occ16File = 'work/maxsup-threshold-selection/contrast-max-16/contrast-max-16-2016_10_14_12:26:16'

logOcc0 = dlmread(occ1File, ' ', 1, 1)
logOcc1 = dlmread(occ1File, ' ', 1, 1)
logOcc2 = dlmread(occ2File, ' ', 1, 1)
logOcc4 = dlmread(occ4File, ' ', 1, 1)
logOcc8 = dlmread(occ8File, ' ', 1, 1)
logOcc16 = dlmread(occ16File, ' ', 1, 1)

plot(logOcc0(:,2),'-o'); hold on
plot(logOcc1(:,2),'-+'); hold on
plot(logOcc2(:,2),'-*'); hold on
plot(logOcc4(:,2),'-v'); hold on
plot(logOcc8(:,2),'-s'); hold on
plot(logOcc16(:,2),'-d'); hold on

ylabel('Validation accuracy')
xlabel('epoch#')
legend({'maxsup 0', 'maxsup 1', 'maxsup 2', 'maxsup 4', 'maxsup 8', 'maxsup 16'})
