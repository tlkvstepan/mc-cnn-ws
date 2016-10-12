close all;

% 1-pix occ
figure;
title('Occlusion threshold selection')
occ0File = 'work/contrast-dprog-0/contrast-dprog-0-2016_10_12_19:14:48'
occ1File = 'work/contrast-dprog-1/contrast-dprog-1-2016_10_12_20:05:40'
occ2File = 'work/contrast-dprog-2/contrast-dprog-2-2016_10_12_21:00:18'
occ4File = 'work/contrast-dprog-4/contrast-dprog-4-2016_10_12_22:04:40'

logOcc0 = dlmread(occ1File, ' ', 1, 1)
logOcc1 = dlmread(occ1File, ' ', 1, 1)
logOcc2 = dlmread(occ2File, ' ', 1, 1)
logOcc4 = dlmread(occ4File, ' ', 1, 1)

plot(logOcc0(:,2)); hold on
plot(logOcc1(:,2)); hold on
plot(logOcc2(:,2)); hold on
plot(logOcc4(:,2)); hold on

ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Occ0', 'Occ1', 'Occ2', 'Occ4'})
