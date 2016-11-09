import main
from bokeh.layouts import gridplot
from bokeh.palettes import Spectral10
from bokeh.plotting import figure, output_file, show

if __name__ == '__main__':
    output_file('plot_rw.html')

    plots = []
    # Random Weights
    for lr in range(10):
        y = [main.neural_net(1000, (lr+1)/10, True, False ) for i in range(10)]
        x = [i for i in range(1000)]
        p = figure(title='Influence of Random weight with learning rate '+str((lr+1)/10), x_axis_label='Learning iterations over the whole sample', y_axis_label='Sum of squared deviation from desired output')
        for n, i in enumerate(y):
            p.line(x, i, line_width = 2, line_color = Spectral10[n])
        plots.append(p)

    show(gridplot(plots, ncols=3, plot_width = 500, plot_height = 300))
