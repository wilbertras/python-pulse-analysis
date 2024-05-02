import tkinter as tk
from tkinter import filedialog
from classes import MKID
import functions as f
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve, find_peaks


class GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Data Visualization GUI")

        self.file_path = ""

        path_label = tk.Label(self.master, text='file path')
        path_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        path_button = tk.Button(self.master, text="Select file", command=self.select_directory)
        path_button.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        smooth_label = tk.Label(self.master, text='smooth window')
        smooth_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.smooth_entry = tk.Entry(self.master)
        self.smooth_entry.insert(0, '')
        self.smooth_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        smooth_unit_label = tk.Label(self.master, text='nr points')
        smooth_unit_label.grid(row=1, column=2, sticky="w", padx=5, pady=5)
        thres_label = tk.Label(self.master, text='threshold')
        thres_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.thres_entry = tk.Entry(self.master)
        self.thres_entry.insert(0, 3)
        self.thres_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        thres_unit_label = tk.Label(self.master, text='stds')
        thres_unit_label.grid(row=2, column=2, sticky="w", padx=5, pady=5)

    def select_directory(self):
        path = filedialog.askopenfilename()
        name = path.split('/')[-1]
        sw = f.ensure_type(self.smooth_entry.get(), int)
        nr_stds = f.ensure_type(self.thres_entry.get(), int)
        if path:
            self.dark_dir = path
            print("file selected:", name)
            amp, theta = f.bin2mat(path)
            _, X = f.smith_coord(theta, amp)
            fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
            ax.plot(X, lw=.5, label='Im(z)', zorder=0)
            max = np.ceil(np.amax(X))
            min = (np.amin(X) // -.5 + 1) *-.5
            ax.set_xlim([0, len(X)])
            ax.set_ylim([min, max])
            if sw:
                window = f.get_window('box', int(2*sw))
                X = convolve(X, window, mode='valid')
                ax.plot(X, lw=.5, label='smoothed Im(z)', zorder=1)
            if nr_stds:
                mph = nr_stds * np.std(X)  
                locs, props = find_peaks(X, height=mph, prominence=mph) 
                heights = props['peak_heights']
                nr_peaks = len(heights)
                ax.scatter(locs, heights, marker='v', c='None', edgecolors='tab:green', lw=1, label='%d peaks' % nr_peaks, zorder=3)
                ax.axhline(mph, color='tab:red', lw=1, label='threshold', zorder=2)
            ax.legend()
            ax.set_xlabel('')
            ax.set_ylabel('Response')
            ax.set_title(name)
            plt.show()


def main():
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
