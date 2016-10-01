
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
logContrastMax = dlmread(logContrastMaxFile, ' ', 1, 1)
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
logMilMaxFile = 'work/mil-contrast-max/mil-contrast-max-2016_09_29_12:58:18'
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
