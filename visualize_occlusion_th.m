close all;

% 1-pix occ
figure;
title('Occlusion threshold selection')
occ1File = 'work/contrast-dprog-1/contrast-dprog-1-2016_10_12_13:35:57'
occ2File = 'work/contrast-dprog-2/contrast-dprog-2-2016_10_12_14:30:59'
occ4File = 'work/contrast-dprog-4/contrast-dprog-4-2016_10_12_15:26:27'
occ8File = 'work/contrast-dprog-8/contrast-dprog-8-2016_10_12_16:20:52'
occ16File = 'work/contrast-dprog-16/contrast-dprog-16-2016_10_12_17:15:16'

logOcc1 = dlmread(occ1File, ' ', 1, 1)
logOcc2 = dlmread(occ2File, ' ', 1, 1)
logOcc4 = dlmread(occ4File, ' ', 1, 1)
logOcc8 = dlmread(occ8File, ' ', 1, 1)
logOcc16 = dlmread(occ16File, ' ', 1, 1)

plot(logOcc1(:,2)); hold on
plot(logOcc2(:,2)); hold on
plot(logOcc4(:,2)); hold on
plot(logOcc8(:,2)); hold on
plot(logOcc16(:,2)); hold on

ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Occ1', 'Occ2', 'Occ4', 'Occ8', 'Occ16'})
