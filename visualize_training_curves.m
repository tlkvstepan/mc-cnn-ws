close all;

% mil-max
figure;
title('mil-max net')
logMilMaxFile = 'work/mil-max/mil-max-2016_09_26_15:08:39'
logMilMax = dlmread(logMilMaxFile, ' ', 1, 1)
yyaxis left
plot(logMilMax(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(logMilMax(:,2)); hold on
plot(smooth(logMilMax(:,2),15)); hold on
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })

% contrast max
figure;
title('contrast-max net')
logContrastMaxFile = 'work/contrast-max/contrast-max-2016_09_28_09:58:58'
logContrastMax = dlmread(logContrastMaxFile, ' ', 1, 1);
yyaxis left
plot(logContrastMax(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(logContrastMax(:,2)); hold on
plot(smooth(logContrastMax(:,2),15)); hold on
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })

% mil-contrast-max
figure;
title('mil-contrast-max net')
logMilContrastMaxFile1 = 'work/mil-contrast-max/mil-contrast-max-2016_09_29_12:58:18'
logMilContrastMaxFile2 =  'work/mil-contrast-max/mil-contrast-max-2016_10_02_13:18:26'
logMilContrastMax = [dlmread(logMilContrastMaxFile1, ' ', 1, 1);...
             dlmread(logMilContrastMaxFile2, ' ', 1, 1)]
yyaxis left
plot(logMilContrastMax(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(logMilContrastMax(:,2)); hold on
plot(smooth(logMilContrastMax(:,2),15)); hold on
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })

% contrast-dprog
figure;
title('contrast-dprog net')
logContrastDprogFile1 = 'work/contrast-dprog/contrast-dprog-2016_10_12_22:53:01'
logContrastDprogFile2 = 'work/contrast-dprog/contrast-dprog-2016_10_14_14:32:52'
logContrastDprog = [dlmread(logContrastDprogFile1, ' ', 1, 1);dlmread(logContrastDprogFile2, ' ', 1, 1)]
yyaxis left
plot(logContrastDprog(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(logContrastDprog(:,2)); hold on
plot(smooth(logContrastDprog(:,2),15)); hold on
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })

% mil-dprog
figure;
title('mil-dprog net')
logFile1 = 'work/mil-dprog/mil-dprog-2016_10_12_22:57:46'
logFile2 = 'work/mil-dprog/mil-dprog-2016_10_13_22:58:17'
logFile3 = 'work/mil-dprog/mil-dprog-2016_10_15_19:21:07'
logFile4 = 'work/mil-dprog/mil-dprog-2016_10_16_12:17:58'
logFile5 = 'work/mil-dprog/mil-dprog-2016_10_17_21:06:35'
logFile6 = 'work/mil-dprog/mil-dprog-2016_10_18_09:54:57'
logFile6 = 'work/mil-dprog/mil-dprog-2016_10_18_09:54:57'
logInfo = [dlmread(logFile1, ' ', 1, 1); dlmread(logFile2, ' ', 1, 1);
           dlmread(logFile3, ' ', 1, 1); dlmread(logFile4, ' ', 1, 1);
           dlmread(logFile5, ' ', 1, 1); dlmread(logFile6, ' ', 1, 1)]
yyaxis left
plot(logInfo(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(logInfo(:,2)); hold on
plot(smooth(logInfo(:,2),15)); hold on
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })

% mil-contrast-dprog
figure;
title('mil-contrast-dprog net')
logFile1 = 'work/mil-contrast-dprog/mil-contrast-dprog-2016_10_13_10:58:04'
logFile2 = 'work/mil-contrast-dprog/mil-contrast-dprog-2016_10_14_12:50:15'
logFile3 = 'work/mil-contrast-dprog/mil-contrast-dprog-2016_10_15_19:16:25'
logFile4 = 'work/mil-contrast-dprog/mil-contrast-dprog-2016_10_19_14:30:19'
logFile5 = 'work/mil-contrast-dprog/mil-contrast-dprog-2016_10_20_11:49:47'
logInfo = [dlmread(logFile1, ' ', 1, 1);
           dlmread(logFile2, ' ', 1, 1);
           dlmread(logFile3, ' ', 1, 1);
           dlmread(logFile4, ' ', 1, 1);
           dlmread(logFile5, ' ', 1, 1)]
yyaxis left
plot(logInfo(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(logInfo(:,2)); hold on
plot(smooth(logInfo(:,2),15)); hold on
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })

