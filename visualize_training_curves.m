close all;

large_scale = true


if large_scale 
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Large scale learning with contrast-dprog KITTI and KITTI15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
file1 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_10_26_17:41:06'
file2 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_10_27_09:48:48'
log = [dlmread(file1, ' ', 1, 1); dlmread(file2, ' ', 1, 1)]
yyaxis left
plot(log(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(log(:,2)); hold on
plot(smooth(log(:,2),15)); hold on
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })
title('Large scale contrast-dprog KITTI15')

figure;
file1 = 'work/contrast-dprog-kitti/contrast-dprog-kitti-2016_10_26_17:41:03'
file2 = 'work/contrast-dprog-kitti/contrast-dprog-kitti-2016_10_27_09:47:32'
log = [dlmread(file1, ' ', 1, 1); dlmread(file2, ' ', 1, 1)]
yyaxis left
plot(log(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(log(:,2)); hold on
plot(smooth(log(:,2),15)); hold on
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })
title('Large scale contrast-dprog KITTI')

end