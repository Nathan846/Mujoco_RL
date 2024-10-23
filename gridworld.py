import numpy as np

rows = 5
columns = 5
states = rows * columns
gamma = 0.9
V = np.zeros(states)
VV = np.zeros((rows, columns))

AA = (1 * columns) + 0
BB = (3 * columns) + 0
AAprime = (1 * columns) + 4
BBprime = (3 * columns) + 2

def setup():
    global rows, columns, states, AA, BB, AAprime, BBprime, V, VV
    rows = 5
    columns = 5
    states = rows * columns
    AA = state_from_xy(1, 0)
    BB = state_from_xy(3, 0)
    AAprime = state_from_xy(1, 4)
    BBprime = state_from_xy(3, 2)
    V = np.zeros(states)
    VV = np.zeros((rows, columns))

def compute_V():
    while True:
        delta = 0
        for x in range(states):
            old_V = V[x]
            V[x] = np.mean([full_backup(x, a) for a in range(4)])
            delta += abs(old_V - V[x])
        if delta < 0.000001:
            break
    update_VV()
    sfa(VV)

def compute_V_star():
    print(VV)
    while True:
        delta = 0
        for x in range(states):
            old_V = V[x]
            V[x] = max([full_backup(x, a) for a in range(4)])
            delta += abs(old_V - V[x])
        if delta < 0.000001:
            break
    update_VV()
    sfa(VV)

def sfa(array):
    if array.ndim == 1:
        for e in array:
            print(f"{e:5.1f}", end=" ")
    else:
        for i in range(array.shape[0]):
            print("\n")
            for j in range(array.shape[1]):
                print(f"{array[i, j]:5.1f}", end=" ")

def full_backup(x, a):
    if x == AA:
        r = 10
        y = AAprime
    elif x == BB:
        r = 5
        y = BBprime
    elif off_grid(x, a):
        r = -1
        y = x
    else:
        r = 0
        y = next_state(x, a)
    return r + (gamma * V[y])

def off_grid(state, a):
    x, y = xy_from_state(state)
    if a == 0 and y + 1 >= rows:
        return True
    elif a == 1 and x + 1 >= columns: 
        return True
    elif a == 2 and y - 1 < 0:
        return True
    elif a == 3 and x - 1 < 0:
        return True
    return False

# Get next state based on action
def next_state(state, a):
    x, y = xy_from_state(state)
    if a == 0: 
        y += 1
    elif a == 1:
        x += 1
    elif a == 2: 
        y -= 1
    elif a == 3:
        x -= 1
    return state_from_xy(x, y)

def state_from_xy(x, y):
    return y + (x * columns)

def xy_from_state(state):
    x = state // columns
    y = state % columns
    return (x, y)

def update_VV():
    for state in range(states):
        x, y = xy_from_state(state)
        VV[y, x] = V[state]

setup()
compute_V()