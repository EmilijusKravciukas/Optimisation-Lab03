import numpy as np

functionCallCount = 0
x0 = np.array([0, 0, 0])
x1 = np.array([1, 1, 1])
xm = np.array([9.0 / 10, 5.0 / 10, 8.0 / 10])

rValues = [10.0, 1.0, 0.1, 0.01]

def OptFunc(x):
    global functionCallCount
    functionCallCount += 1
    return -x[0] * -x[1] * -x[2]

def Equality(x):
    return 2 * (x[0]*x[2] + x[0]*x[1] + x[1]*x[2]) - 1

def Inequality(x):
    return np.negative(x)

def constraintsPenalty(x):
    inequalityPen = 0

    for value in Inequality(x):
        inequalityPen += max(value, 0.0)**2

    equalityPen = Equality(x)**2

    return inequalityPen + equalityPen

def PenalizedFunc(func, x, r):
    return func(x) + 1.0/r * constraintsPenalty(x)

def NumericalGradient(func, x):
    gamma = 1e-6
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_forward = np.copy(x)
        x_backward = np.copy(x)
        x_forward[i] += gamma
        x_backward[i] -= gamma
        grad[i] = (func(x_forward) - func(x_backward)) / (2 * gamma)
    return grad

def GradientDescent(func, xStart, gamma = 0.001, tol=1e-4):
    iterationCount = 0
    global functionCallCount

    x = np.copy(xStart)

    gradient = np.array(NumericalGradient(func, x))
    
    while np.linalg.norm(gradient) >= tol:
        x = x - gamma * gradient
        gradient = np.array(NumericalGradient(func, x))
        iterationCount += 1

    gradient = np.array(NumericalGradient(func, x))
    print(f"GD Iteration {iterationCount}: x = {x}, gradient norm = {np.linalg.norm(gradient)}")
    print(f"Gradient Descent completed:")
    print(f" Solution: {x}")
    print(f" Function value: {func(x)}")
    print(f" Function calls: {functionCallCount}")
    print(f" Iterations: {iterationCount}")
    functionCallCount = 0
    return x

currentX = np.copy(xm)

for r in rValues:
    passedPenFunc = lambda x : PenalizedFunc(OptFunc, x, r)
    currentX = GradientDescent(passedPenFunc, currentX)

