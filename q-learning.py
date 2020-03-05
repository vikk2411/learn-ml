import numpy as np
import matplotlib.pylab as plt
import networkx as nx


points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]

G=nx.Graph()
G.add_edges_from(points_list)
pos = nx.spring_layout(G)
# print("pos---> ", G.nodes)
nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
# plt.show()

# how many points in graph? x points
MATRIX_SIZE = 8

# create matrix x*y
R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
R *= -1
print(R)
# shape gives the size of the nxn matrix

goal = 6
# add goal point round trip
R[goal,goal]= 100

# assign zeros to paths and 100 to goal-reaching point
for point in points_list:
  if point[1] == goal:
    R[point] = 100
  else:
    R[point] = 0

  if point[0] == goal:
    R[point[::-1]] = 100  # reverse of the point
  else:
    R[point[::-1]] = 0

print(R)

Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))
gamma = 0.8 # learning rate

initial_state = 1

def available_points(state):
  current_row = R[state,] # returns the full row of the matrix
  return np.where(current_row >= 0)[1]

# print(available_points(7))
available_actions = available_points(initial_state)

def next_sample_state(available_actions):
  return int(np.random.choice(available_actions, 1))

next_state = next_sample_state(available_actions)

def update(current_state, actions, gamma):
  max_index = np.where(Q[actions,] == np.max(Q[actions,]))[1]

  if max_index.shape[0] > 1:
    max_index = int(np.random.choice(max_index, size = 1))
  else:
    max_index = int(max_index)

  max_value = Q[actions, max_index]

  Q[current_state, actions] = R[current_state, actions] + gamma * max_value
  # print('max_value', R[current_state, actions] + gamma * max_value)

  if (np.max(Q) > 0):
    return(np.sum(Q/np.max(Q)*100))
  else:
    return (0)

# update(initial_state, available_actions, gamma)

scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_points(current_state)
    actions = next_sample_state(available_act)
    score = update(current_state,actions,gamma)
    scores.append(score)
    # print ('Score:', str(score))

print("Trained Q matrix:")
print(Q/np.max(Q)*100)


# Testing
current_state = 0
steps = [current_state]

while current_state != goal:
  next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

  if next_step_index.shape[0] > 1:
    next_step_index = int(np.random.choice(next_step_index, size = 1))
  else:
    next_step_index = int(next_step_index)

  steps.append(next_step_index)
  current_state = next_step_index

print("Most efficient path:")
print(steps)

plt.plot(scores)
plt.show()