close all;

% 1-pix occ
figure;
title('Occlusion threshold selection')
occ1File = 'work/contrast-dprog/contrast-dprog-2016_10_04_14:31:25'
occ8File = 'work/contrast-dprog-occ8/contrast-dprog-occ8-2016_10_07_18:34:41'

logOcc1 = dlmread(occ1File, ' ', 1, 1)
logOcc8 = dlmread(occ8File, ' ', 1, 1)


plot(logOcc1(:,2)); hold on

plot(logOcc8(:,2)); hold on
%plot(logOcc16(:,2)); hold on
%plot(logOcc32(:,2)); hold on

ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Occ1', 'Occ8'})
