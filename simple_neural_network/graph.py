import main
from bokeh.layouts import layout
from bokeh.palettes import Spectral10
from bokeh.plotting import figure, output_file, show

if __name__ == '__main__':
    learning_rate = 1.0
    output_file('plot.html')

    # Random Weights

    y = [main.neural_net(1000, learning_rate, True, False ) for i in range(10)]
    x = [i for i in range(1000)]

    p1 = figure(title='Influence of Random weight', x_axis_label='Learning iterations over the whole sample', y_axis_label='Sum of squared deviation from desired output')

    for n, i in enumerate(y):
        p1.line(x, i, legend = 'Random Weight '+str(n), line_width = 2, line_color = Spectral10[n])

    # Shuffle Input

    y = [main.neural_net(1000, learning_rate, False, True) for i in range(10)]

    p2 = figure(title='Influence of shuffled learning examples', x_axis_label='Learning iterations over the whole sample', y_axis_label='Sum of squared deviation from desired output')

    for n, i in enumerate(y):
        if n == 0:
            p2.line(x, main.neural_net(1000, 0.5, False, False), legend = 'No Shuffle', line_width = 2, line_color = Spectral10[n])
        else:
            p2.line(x, i, legend = 'Shuffled Input '+str(n-1), line_width = 2, line_color=Spectral10[n])

    # Different Learning Rates

    y = [main.neural_net(3000, (i+1)/10, True, False) for i in range(10)]
    x = [i for i in range(3000)]

    p3 = figure(title='Influence of different learning rates', x_axis_label='Learning iterations over the whole sample', y_axis_label='Sum of squared deviation from desired output')

    for n, i in enumerate(y):
        p3.line(x, i, legend = 'Learning rate '+str((n+1)/10), line_width = 2, line_color = Spectral10[n])

    show(layout([[p1, p2],[p3]], sizing_mode='stretch_both'))
