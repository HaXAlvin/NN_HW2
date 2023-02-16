from simple_playground import run_example, Playground
from model import RBF
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys


def draw_border():
    for line in p.lines: # draw road
        ax.plot([line.p1.x,line.p2.x],[line.p1.y,line.p2.y],c='r')

def reset_all():
    global p
    p = Playground()
    ax.cla()
    draw_border()
    canvas.draw()
    canvas.flush_events()

def draw_car():
    global p
    if features.get() == 4:
        rbf = RBF(2, focus_count.get(), 1)
        rbf.train(four_data, four_target)
    else:
        rbf = RBF(4, focus_count.get(), 1)
        rbf.train(six_data, six_target)
    p = Playground(rbf)

    path = run_example(p)
    ax.plot(path['x'],path['y'],alpha=0.8,label=f"{features.get()}D-{focus_count.get()}") # draw car go path
    draw_border()
    ax.legend()
    canvas.draw()
    canvas.flush_events()

def radiobutton_event():
    if features.get() == 4:
        focus_count.value = 800
    else:
        focus_count.value = 1000
    focus_count_input.delete(0, tk.END)
    focus_count_input.insert(0, focus_count.value)

if __name__ == '__main__':
    six_data = np.loadtxt("./train6dAll.txt")
    six_target = six_data[:,5]
    six_data = np.stack([six_data[:,0],six_data[:,1],six_data[:,2],six_data[:,4]-six_data[:,3]]).T


    four_data = np.loadtxt("./train4dAll.txt")
    four_target = four_data[:,3]
    four_data = np.stack([four_data[:,0],four_data[:,2]-four_data[:,1]]).T

    p = Playground()

    root = tk.Tk()
    root.title("RBF Playground")
    root.protocol("WM_DELETE_WINDOW", lambda : sys.exit(0))

    
    features = tk.IntVar(value=4)
    focus_count = tk.IntVar(value=800)

    feature_label = tk.Label(root, text="Feature Count:")
    feature_label.grid(row=0, column=0)
    radio_4_button = tk.Radiobutton(root, text="4D", value=4, variable=features, command=radiobutton_event)
    radio_4_button.select()
    radio_4_button.grid(row=0, column=1)
    radio_6_button = tk.Radiobutton(root, text="6D", value=6, variable=features, command=radiobutton_event)
    radio_6_button.grid(row=0, column=2)

    focus_count_label = tk.Label(root, text="Focus Count:")
    focus_count_label.grid(row=1, column=0)
    focus_count_input = tk.Entry(root, textvariable=focus_count)
    focus_count_input.grid(row=1, column=1, columnspan=2)

    run_button = tk.Button(root, text="Run",command=draw_car)
    run_button.grid(row=2, column=0, columnspan=2)

    reset_all_button = tk.Button(root, text="Reset All",command=reset_all)
    reset_all_button.grid(row=2, column=2)

    fig = plt.Figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111)
    draw_border()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=3, column=0, columnspan=3)


    root.mainloop()
