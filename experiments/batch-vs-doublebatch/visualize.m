
result_batch = dlmread('work/batch_learning.log', ' ', 1, 1)
result_doubleBatch = dlmread('work/doubleBatch_learning.log', ' ', 1, 1)

yyaxis left
plot(result_batch(:,1)); hold on
plot(result_doubleBatch(:,1)); hold on
ylabel('Train error max(0, -x^{+}+x^{-}+\mu)')
yyaxis right
plot(result_batch(:,2)); hold on
plot(result_doubleBatch(:,2));
ylabel('Test error N(x^{+} > x^{-})/N')
xlabel('epoch# (256x5 epipolar lines each)')
legend({'batch', 'doubleBatch'})