close all;

figure;
title('mil-max net')
logFile1 = 'work/TEST-mc-cnn/TEST-mc-cnn_plotErr'
log1 = dlmread(logFile1)
logFile2 = 'work/TEST-contrast-dprog/TEST-contrast-dprog_plotErr'
log2 = dlmread(logFile2)

plot(log1, '-+'); hold on
plot(log2, '-x');

ylabel('% of GT disparities')
xlabel('Number of maximums')
legend({'mc-cnn', 'contrast-dprog'})

