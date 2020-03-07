from matplotlib.pyplot import scatter, show

with open("./clustered_hr_data.csv") as file:
    contents = file.read()

x_vals = []
y_vals = []
cats = []

for i, line in enumerate(contents.split('\n')):
    if i == 0 or line == '':
        continue
    parts = line.split(',')
    x_vals.append(float(parts[0]))
    y_vals.append(float(parts[1]))
    cats.append(int(parts[2]))

scatter(x_vals, y_vals, c=cats)
show()