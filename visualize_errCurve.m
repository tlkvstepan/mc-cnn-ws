close all;

figure;
title('mil-max net')
logFile1 = '/HDD1/Dropbox/Research/01_code/mil-mc-cnn/work/test-contrast-max/test-contrast-maxplotErr'
log1 = dlmread(logFile1, ' ', 1, 1)
plot(log1(:,1)); hold on

ylabel('Train error: max(0, -x^{(+)}+x^{(-)}+\mu)')
ylabel('Test accuracy: % of pixels |d-d_{gt}|<3')
xlabel('epoch#')
legend({'Train error', 'Test accuracy', 'Smoothed test accuracy' })
