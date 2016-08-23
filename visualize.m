
result = dlmread('work/learning.log', ' ', 1, 1)

plot(result(:,1));
xlabel('epoch# (320k epipolar lines each)')
ylabel('max(0, - x^{(+)} + x^{(-)} + \mu)')


figure
plot(result(:,2));
xlabel('epoch# (320k epipolar lines each)')
ylabel(' N(x^{(+)} > x^{(-)}) / N')
