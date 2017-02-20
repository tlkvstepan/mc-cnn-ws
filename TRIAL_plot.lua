require 'gnuplot'

file = 'work/test-2017_02_20_11:30:34/err_test-2017_02_20_11:30:34_2017_02_20_11:30:34.txt'


--gnuplot.raw('set autoscale ')

gnuplot.raw('set ylabel "train loss"')
gnuplot.raw('set y2label "valid. err, [%]"')

gnuplot.raw('set xlabel "iter"')
gnuplot.raw('set x1tic auto')
gnuplot.raw('set y1tic auto')
gnuplot.raw('set y2tic auto')

gnuplot.raw('set logscale x')
gnuplot.raw('set logscale y')
gnuplot.raw('set logscale y2')

gnuplot.raw("plot '" .. file .. "' using ($1):(($4))  title 'valid. err' axes x1y2, '"
                     .. file .. "' using ($1):(($3))  title 'train. loss' axes x1y1")


