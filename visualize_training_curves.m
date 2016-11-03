close all;

large_scale = true
normal_scale = true

if large_scale 
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Large scale learning with contrast-dprog KITTI and KITTI15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure;
file1 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_10_31_10:35:58'
file2 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_11_01_12:13:42'
file3 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_11_02_12:40:20'

log = [dlmread(file1, ' ', 1, 1);dlmread(file2, ' ', 1, 1);dlmread(file3, ' ', 1, 1)]
yyaxis left
plot(log(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(log(:,2)); hold on
plot(smooth(log(:,2),15)); hold on
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })
title('Large-scale contrast-dprog KITTI15')


figure;
file1 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_10_31_14:27:29'
file2 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_01_14:44:13'
file3 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_02_17:11:08'
log = [dlmread(file1, ' ', 1, 1); dlmread(file2, ' ', 1, 1); dlmread(file3, ' ', 1, 1)]
yyaxis left
plot(log(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(log(:,2)); hold on
plot(smooth(log(:,2),15)); hold on
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })
title('Large-scale contrast-dprog KITTI')


end

if normal_scale 
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Normal learning with contrast-dprog KITTI and KITTI15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
file1 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_10_31_10:05:40'
file2 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_11_01_10:25:54'
file3 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_11_02_12:39:07'
log = [dlmread(file1, ' ', 1, 1); dlmread(file2, ' ', 1, 1); dlmread(file3, ' ', 1, 1)]
yyaxis left
plot(log(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(log(:,2)); hold on
plot(smooth(log(:,2),15)); hold on
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })
title('Contrast-dprog KITTI15')

figure;
file1 = 'work/contrast-dprog/contrast-dprog-2016_10_12_22:53:01'
file2 = 'work/contrast-dprog/contrast-dprog-2016_10_14_14:32:52'
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
title('Contrast-dprog KITTI')

figure;
file1 = 'work/contrast-dprog-mb/contrast-dprog-mb-2016_11_01_17:44:25'
file2 = 'work/contrast-dprog-mb/contrast-dprog-mb-2016_11_02_19:05:06'
log = [dlmread(file1, ' ', 1, 1);dlmread(file2, ' ', 1, 1)]
yyaxis left
plot(log(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(log(:,2)); hold on
plot(smooth(log(:,2),15)); hold on
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })
title('Contrast-dprog MB')

end