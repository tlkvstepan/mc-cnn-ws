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
logContrastDprogFile = 'work/contrast-dprog/contrast-dprog-2016_10_12_22:53:01'
logContrastDprog = dlmread(logContrastDprogFile, ' ', 1, 1)
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
logInfo = [dlmread(logFile1, ' ', 1, 1);dlmread(logFile2, ' ', 1, 1)]
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
logInfo = dlmread(logFile1, ' ', 1, 1);
yyaxis left
plot(logInfo(:,1)); hold on
ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(logInfo(:,2)); hold on
plot(smooth(logInfo(:,2),15)); hold on
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })

