import json
from bokeh.plotting import figure, output_file, show

with open('annotation_bigsick.json', 'r') as file:
    data = json.load(file)
print(data)
output_file('people.html')

plot = figure(plot_width=800, plot_height=400)

x = [frame['count']*24 for frame in data['frames']]
y = [frame['time'] for frame in data['frames']]


print(x, y)
plot.line(y, x, line_width=2)

show(plot)
