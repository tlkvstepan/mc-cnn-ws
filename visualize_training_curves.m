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
file4 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_11_03_13:25:51'
file5 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_11_04_15:06:00'
file6 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_11_05_17:46:34'
file7 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_11_06_13:03:41'
file8 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_11_07_10:29:22'
file9 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_11_08_14:40:01'
file10 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_11_09_13:23:53'
file11 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_11_10_13:37:59'
file12 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_11_11_13:28:33'
file13 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_11_12_16:33:36'
file14 = 'work/contrast-dprog-kitti15-ext/contrast-dprog-kitti15-ext-2016_11_13_15:25:55'

log = [ dlmread(file1, ' ', 1, 1);dlmread(file2, ' ', 1, 1);
        dlmread(file3, ' ', 1, 1);dlmread(file4, ' ', 1, 1);
        dlmread(file5, ' ', 1, 1);dlmread(file6, ' ', 1, 1);
        dlmread(file7, ' ', 1, 1);dlmread(file8, ' ', 1, 1);
        dlmread(file9, ' ', 1, 1);dlmread(file10, ' ', 1, 1);
        dlmread(file11, ' ', 1, 1);dlmread(file12, ' ', 1, 1);
        dlmread(file13, ' ', 1, 1);dlmread(file14, ' ', 1, 1);]
    
yyaxis left
loglog(log(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
loglog(100-log(:,2)); hold on
loglog(100-smooth(log(:,2),15)); hold on
ylabel('Test error: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'WTA error', 'Smoothed WTA error' })
title('Large-scale contrast-dprog KITTI15')


figure;

file1 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_10_31_14:27:29'
file2 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_01_14:44:13'
file3 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_02_17:11:08'
file4 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_03_21:52:00'
file5 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_05_14:29:25'
file6 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_06_13:04:31'
file7 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_07_10:26:32'
file8 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_08_14:38:08'
file9 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_09_13:23:53'
file10 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_10_13:39:03'
file11 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_11_13:27:36'
file12 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_12_16:34:44'
file13 = 'work/contrast-dprog-kitti-ext/contrast-dprog-kitti-ext-2016_11_13_15:24:53'

log = [dlmread(file1, ' ', 1, 1); dlmread(file2, ' ', 1, 1); 
       dlmread(file3, ' ', 1, 1); dlmread(file4, ' ', 1, 1); 
       dlmread(file5, ' ', 1, 1); dlmread(file6, ' ', 1, 1);
       dlmread(file7, ' ', 1, 1); dlmread(file8, ' ', 1, 1);
       dlmread(file9, ' ', 1, 1);dlmread(file10, ' ', 1, 1);
       dlmread(file11, ' ', 1, 1);dlmread(file12, ' ', 1, 1);dlmread(file13, ' ', 1, 1)]
yyaxis left
loglog(log(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
loglog(100-log(:,2)); hold on
loglog(100-smooth(log(:,2),15)); hold on
ylabel('Test error: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'WTA error', 'Smoothed WTA error' })
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
file4 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_11_03_13:24:23'
file5 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_11_04_15:12:59'
file6 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_11_05_17:44:19'
file7 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_11_06_13:05:34'
file8 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_11_08_14:36:05'
file9 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_11_09_13:23:48'
file10 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_11_10_13:36:08'
file11 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_11_11_13:26:17'
file12 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_11_12_16:35:52'
file13 = 'work/contrast-dprog-kitti15/contrast-dprog-kitti15-2016_11_13_15:23:50'

log = [dlmread(file1, ' ', 1, 1); dlmread(file2, ' ', 1, 1);
       dlmread(file3, ' ', 1, 1); dlmread(file4, ' ', 1, 1);
       dlmread(file5, ' ', 1, 1); dlmread(file6, ' ', 1, 1);
       dlmread(file7, ' ', 1, 1); dlmread(file8, ' ', 1, 1);
       dlmread(file9, ' ', 1, 1); dlmread(file10, ' ', 1, 1);
       dlmread(file11, ' ', 1, 1); dlmread(file12, ' ', 1, 1); dlmread(file13, ' ', 1, 1);]
yyaxis left
loglog(log(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
loglog(100-log(:,2)); hold on
loglog(100-smooth(log(:,2),15)); hold on
ylabel('Test error: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'WTA error', 'Smoothed WTA error' })
title('Contrast-dprog KITTI15')

figure;
file1 = 'work/contrast-dprog-kitti/contrast-dprog-2016_10_12_22:53:01'
file2 = 'work/contrast-dprog-kitti/contrast-dprog-2016_10_14_14:32:52'
file3 = 'work/contrast-dprog-kitti/contrast-dprog-kitti-2016_11_08_14:33:41'
file4 = 'work/contrast-dprog-kitti/contrast-dprog-kitti-2016_11_09_13:23:53'
file5 = 'work/contrast-dprog-kitti/contrast-dprog-kitti-2016_11_10_13:35:02'
file6 = 'work/contrast-dprog-kitti/contrast-dprog-kitti-2016_11_11_13:25:48'
file7 = 'work/contrast-dprog-kitti/contrast-dprog-kitti-2016_11_12_16:31:02'
file8 = 'work/contrast-dprog-kitti/contrast-dprog-kitti-2016_11_13_15:27:45'

log = [dlmread(file1, ' ', 1, 1); dlmread(file2, ' ', 1, 1); 
       dlmread(file3, ' ', 1, 1); dlmread(file4, ' ', 1, 1);
       dlmread(file5, ' ', 1, 1); dlmread(file6, ' ', 1, 1);
       dlmread(file7, ' ', 1, 1);dlmread(file8, ' ', 1, 1)]
yyaxis left
loglog(log(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
loglog(100-log(:,2)); hold on
loglog(100-smooth(log(:,2),15)); hold on
ylabel('Test error: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'WTA error', 'Smoothed WTA error' })
title('Contrast-dprog KITTI')

figure;
file1 = 'work/contrast-dprog-mb/contrast-dprog-mb-2016_11_01_17:44:25'
file2 = 'work/contrast-dprog-mb/contrast-dprog-mb-2016_11_02_19:05:06'
file3 = 'work/contrast-dprog-mb/contrast-dprog-mb-2016_11_07_13:56:18'
file4 = 'work/contrast-dprog-mb/contrast-dprog-mb-2016_11_08_14:41:36'
file5 = 'work/contrast-dprog-mb/contrast-dprog-mb-2016_11_09_13:24:19'
file6 = 'work/contrast-dprog-mb/contrast-dprog-mb-2016_11_10_13:37:43'
file7 = 'work/contrast-dprog-mb/contrast-dprog-mb-2016_11_11_13:29:59'
file8 = 'work/contrast-dprog-mb/contrast-dprog-mb-2016_11_12_16:33:03'
file9 = 'work/contrast-dprog-mb/contrast-dprog-mb-2016_11_13_15:27:26'

log = [dlmread(file1, ' ', 1, 1);dlmread(file2, ' ', 1, 1)
       dlmread(file3, ' ', 1, 1);dlmread(file4, ' ', 1, 1);
       dlmread(file5, ' ', 1, 1);dlmread(file6, ' ', 1, 1);
       dlmread(file7, ' ', 1, 1);dlmread(file8, ' ', 1, 1);
       dlmread(file9, ' ', 1, 1)]
yyaxis left
loglog(log(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
loglog(100*log(:,2)); hold on
loglog(100*smooth(log(:,2),15)); hold on
ylabel('Test error: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'WTA error', 'Smoothed WTA error' })
title('Contrast-dprog MB')

end