from matplotlib.pyplot import scatter, show
from sys import argv

# with open("./clustered_hr_data.csv") as file:
if len(argv) != 2:
    print("No filename given")
    quit()
with open(argv[1]) as file:
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
