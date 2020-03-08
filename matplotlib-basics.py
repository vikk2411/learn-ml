import matplotlib.pyplot as plt
import numpy as np


x = np.arange(1,11)
y = 2 * x
yn = x - 3


# plt.plot(x,y, color="orange", linewidth=2) #color and width are optional
# plt.plot(x,yn, linestyle=":") #color and width are optional
# plt.title("Simple Line")
# plt.xlabel(" X Axis ")
# plt.ylabel(" Y Axis ")
# plt.grid(True)  #default is False
# plt.show()


# lets plt a bar-graph

students = { "Sam": 30, "Alice": 32, "Mark": 40 }
# print(students.keys()) # returns dict_keys(['Sam', 'Alice', 'Mark'])
names = list(students.keys())
marks = list(students.values())

# barh for horizontal graph
plt.bar(names,marks)
plt.show()

# similar for scatter plot with scatter()
# similar for histogram with hist()