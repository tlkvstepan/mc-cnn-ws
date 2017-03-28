
result_batch = dlmread('/HDD1/Dropbox/Research/01_code/mil-mc-cnn/work/TRAIN_MIL_ACRT_KITTI_KITTI_EXT/err_TRAIN_MIL_ACRT_KITTI_KITTI_EXT_2017_02_13_18:42:24.txt')

yyaxis left
plot(result_batch(:,3)); hold on
ylabel('Train loss: max(0, -x^{(+)}+x^{(-)}+\mu)')
yyaxis right
plot(result_batch(:,4)); hold on
ylabel('Test error: N(x^{(+)}>x^{(-)})/N')
xlabel('epoch# (370x100 epipolar lines each)')
legend({'MIL-ACRT-ARCH-KITTI'})