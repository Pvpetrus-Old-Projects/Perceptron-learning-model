import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageTk
from PIL import Image as img
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn import datasets
from tkinter import *
from random import randrange
import time
import threading

thread = None
def _unit_step_func(x):
    return np.where(x >= 0, 1, 0)


class Perceptron:
    def __init__(self, learning_rate=0.01, number_of_iterations=1000):
        self.y_np = None
        self.errors_ = []
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.activation_function = _unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.y_np = np.array([1 if i > 0 else 0 for i in y])
        self.errors_ = []
        for i in range(self.number_of_iterations):
            errors = 0
            for index, x_index in enumerate(X):
                update = self.learning_rate * (
                        self.y_np[index] - self.activation_function(np.dot(x_index, self.weights) + self.bias))
                self.weights += update * x_index
                self.bias += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        Create_Plot(X, self.y_np, p)
        Create_Error_Plot()
        time.sleep(0.25)
        return self

    def predict(self, X):
        return self.activation_function(np.dot(X, self.weights) + self.bias)

    # def net_input(self, X):
    #   return np.dot(X, self.weights[1:]) + self.weights[0]


X, y = datasets.make_blobs(
    n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=123
)

p = Perceptron(learning_rate=0.01, number_of_iterations=0)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def Create_Error_Plot():
    plt.close()
    plt.cla()
    plt.clf()
    fig = plt.figure()
    plt.plot(range(1, len(p.errors_) + 1), p.errors_, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Misclassifications')
    canvas = FigureCanvasTkAgg(fig)
    canvas.draw()
    canvas.get_tk_widget().grid(row=2, column=0)


def Create_Plot(X_train, y_train, p):
    plt.close()
    plt.cla()
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
    if p.number_of_iterations != 0:
        x0_1 = np.amin(X_train[:, 0])
        x0_2 = np.amax(X_train[:, 0])
        x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
        x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]
        ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")
    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])
    canvas = FigureCanvasTkAgg(fig)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=0)


def ShowInputsAndWeights():
    iteration = 50
    for input_, weight in X_train:
        Label(root, text="input: %f, weight: %f" % (input_, weight), fg="blue", bg="yellow"). \
            place(x=1100, y=iteration)  # space between
        iteration += 50


def PerceptronLearningRate():
    iteration = 50
    # for weight in p.weights:
    #   Label(root, text="weight: %f" % weight, fg="blue", bg="yellow").place(x=900, y=iteration)  # space between
    #  iteration += 50
    Label(root, text="Learnign rate: %f" % p.learning_rate, fg="blue", bg="yellow").place(x=900, y=iteration)


def RestartButtonFunction():
    listOfGlobals = globals()
    listOfGlobals["X"], listOfGlobals["y"] = datasets.make_blobs(
        n_samples=randrange(100) + 50, n_features=2, centers=randrange(3) + 2, cluster_std=1.05, random_state=2
    )
    listOfGlobals["X_train"], listOfGlobals["X_test"], listOfGlobals["y_train"], listOfGlobals[
        "y_test"] = train_test_split(
        listOfGlobals["X"], listOfGlobals["y"], test_size=0.33, random_state=123
    )
    p.y_np = None
    p.errors_ = []
    p.learning_rate = 0.01
    p.number_of_iterations = 0
    p.weights = None
    p.bias = None
    p.fit(X_train, y_train)
    # zmiana
    p.predict(X_test)
    UpdateWeights()


def OneIterationButtonFunction():
    p.number_of_iterations += 1
    p.fit(X_train, y_train)
    # zmiana
    p.predict(X_test)
    PerceptronLearningRate()
    UpdateWeights()


def TenIterationButtonFunction():
    for i in range(0, 10):
        p.number_of_iterations += 1
        p.fit(X_train, y_train)
        p.predict(X_test)
        PerceptronLearningRate()
        UpdateWeights()


def OneHundredIterationButtonFunction():
    for i in range(0, 100):
        p.number_of_iterations += 1
        p.fit(X_train, y_train)
        p.predict(X_test)
        PerceptronLearningRate()
        UpdateWeights()


def LearningRateFunction():
    try:
        p.learning_rate = float(InputLearningRate.get())
        PerceptronLearningRate()
    except ValueError:
        print("Nie podano liczby")


def UpdateWeights():
    index = 0
    space = 0
    for weight in p.weights:
        Label(image=SingleInputImage).place(x=1200, y=100 + space)
        Label(text="%f" % weight).place(x=1280, y=130 + space)
        Label(text="x%d" % index).place(x=1230, y=130 + space)
        index += 1
        space += 100


# tkinter window app

root = Tk()
root.title("Perceptron Learning")
root.geometry("1500x1000")
# text

Label(root, text="Perceptron learning").grid(row=0, column=0)
Label(root, text="Options").grid(row=0, column=2)


# button functions

def TenIterationButtonFunctionThread():

    global thread
    thread = threading.Thread(TenIterationButtonFunction())
    thread.start()
    thread.join()

def OneHundredIterationButtonFunctionThread():
    global thread
    thread = threading.Thread(OneHundredIterationButtonFunction())
    thread.start()
    thread.join()

# buttons

Button(root, text="Restart", command=RestartButtonFunction, width=15, fg="blue", bg="yellow", padx=50).place(x=680,
                                                                                                             y=700)
Button(root, text="1 iteration", command=OneIterationButtonFunction, width=15, fg="blue", bg="yellow", padx=50).place(
    x=680, y=200)
Button(root, text="10 iterations",
       command=TenIterationButtonFunctionThread, width=15,
       fg="blue", bg="yellow",
       padx=50).place(x=680, y=300)
# Button(root, text="10 iterations", command=TenIterationButtonFunction, width=15, fg="blue", bg="yellow",
#       padx=50).place(x=680, y=300)
Button(root, text="100 iterations", command=OneHundredIterationButtonFunctionThread, width=15, fg="blue", bg="yellow",
       padx=50).place(x=680, y=400)
Button(root, text="Confirm learning rate input:", command=LearningRateFunction, width=15, fg="blue", bg="yellow",
       padx=50).place(x=680, y=500)
Button(root, text="Exit program", command=root.quit).grid(row=0, column=3)

# input

InputLearningRate = Entry(root, width=30, borderwidth=5)
InputLearningRate.insert(0, "Enter a number")
InputLearningRate.place(x=680, y=600)

# spaces between elements
Label(root, text="             ").grid(row=0, column=1)  # space between
Label(root, text="             ").grid(row=1, column=2)  # space between
Label(root, text="             ").grid(row=3, column=2)  # space between
Label(root, text="             ").grid(row=5, column=2)  # space between
Label(root, text="             ").grid(row=7, column=2)  # space between
Label(root, text="             ").grid(row=9, column=2)  # space between
# loop

# images

PerceptronImage = ImageTk.PhotoImage(img.open("Perceptron.jpg"))
Label(image=PerceptronImage).place(x=1150, y=50)

SingleInputImage = ImageTk.PhotoImage(img.open("Input.jpg"))

p.fit(X_train, y_train)
predictions = p.predict(X_test)

# ShowInputsAndWeights()

# Create_Error_Plot()

"""print("Perceptron classification accuracy", accuracy(y_test, predictions))
print("Inputs: ", X)
print("y: ", y)
print("p.y_np: ", p.y_np)
print("weights: ", p.weights)"""
# creating a plot
print(X, y)

# X - elements of a table (lines) they have n_features attributes
# y - decision class
# weights - weights of inputs that are updated
# errors - amount of bad predictions

root.mainloop()

# perceptron
