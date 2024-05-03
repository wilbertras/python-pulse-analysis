import tkinter as tk
from tkinter import filedialog
from classes import MKID
import functions as f
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve, find_peaks
from glob import glob
import re


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
        nrfiles_label = tk.Label(self.master, text='nr of files')
        nrfiles_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.nrfiles_entry = tk.Entry(self.master)
        self.nrfiles_entry.insert(0, 1)
        self.nrfiles_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        window_label = tk.Label(self.master, text='pulse window')
        window_label.grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.window_entry = tk.Entry(self.master)
        self.window_entry.insert(0, 200)
        self.window_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        window_unit_label = tk.Label(self.master, text='nr points')
        window_unit_label.grid(row=4, column=2, sticky="w", padx=5, pady=5)


    def select_directory(self):
        path = filedialog.askopenfilename()
        sw = f.ensure_type(self.smooth_entry.get(), int)
        nr_stds = f.ensure_type(self.thres_entry.get(), int)
        nr_files = f.ensure_type(self.nrfiles_entry.get(), int)
        pw = f.ensure_type(self.window_entry.get(), int)
        if path:
            name = path.split('/')[-1]
            if nr_files > 1:
                dir = '/'.join(path.split('/')[:-1]) 
                base = '_'.join(name.split('_')[:2])
                files = glob(dir + '/' + base + '__TDvis' + '*.bin')
                files = sorted(files, key=lambda s: int(re.findall(r'\d+', s)[-2]))
                max_nr_files = len(files)
                if nr_files > max_nr_files:
                    nr_files = max_nr_files
                amp, theta, _ = f.concat_vis(files[:nr_files])
                name = files[0].split('/')[-1]
            else:
                self.dark_dir = path
                amp, theta = f.bin2mat(path)
            
            _, X = f.smith_coord(theta, amp)
            nr_points = len(X)
            plot_X = int(nr_points / nr_files)
            max = np.ceil(np.amax(X[:plot_X]))
            min = (np.amin(X[:plot_X]) // -.5 + 1) *-.5
            
            fig, axes = plt.subplot_mosaic('aabc', figsize=(12, 3), constrained_layout=True, sharey=True)
            ax = axes['a']
            ax.plot(X[:plot_X], lw=.5, label='Im(z)', zorder=0)
            ax.set_xlim([0, plot_X])
            ax.set_ylim([min, max])
            ax.set_xlabel('')
            ax.set_ylabel('Response')
            ax.set_title(name)
            if sw:
                window = f.get_window('box', int(2*sw))
                Xsmooth = convolve(X, window, mode='valid')
                ax.plot(Xsmooth[:plot_X], lw=.5, label='smoothed Im(z)', zorder=1)
            if nr_stds:
                if sw:
                    mph = nr_stds * np.std(Xsmooth)
                    locs, props = find_peaks(Xsmooth, height=mph, prominence=mph) 
                    heights = X[locs]
                else:
                    mph = nr_stds * np.std(X)
                    locs, props = find_peaks(X, height=mph, prominence=mph) 
                    heights = props['peak_heights']
                nr_peaks = len(heights)
                peak_rate = nr_peaks / nr_files
                plot_locs = locs < plot_X
                ax.scatter(locs[plot_locs], heights[plot_locs], marker='v', c='None', edgecolors='tab:green', lw=1, label='peaks', zorder=3)
                ax.axhline(mph, color='tab:red', lw=1, label='threshold', zorder=2)
            ax.legend()
            ax = axes['b']
            ax.hist(heights, 'auto', facecolor='tab:green', label='Nph=%.f cps' % peak_rate, orientation=u'horizontal')
            ax.set_xlabel('Counts')
            ax.legend()
            ax.set_title('Heights vis%d-%d' % (0, nr_files))
            pulses = []
            offset = 10
            for i, loc in enumerate(locs):
                too_close = 0
                if  i < nr_peaks-1 and loc + pw + offset >= locs[i+1] and loc + pw <= nr_points and loc - offset >= 0:
                    too_close += 1
                else:
                    pulses.append(X[loc-offset:loc+pw])
            pulses = np.array(pulses).reshape((-1, pw+offset))
            mean_pulse = np.mean(pulses, axis=0)
            ax = axes['c']
            ax.plot(mean_pulse)
            inset = ax.inset_axes([.5, .5, .45, .45])
            inset.semilogy(mean_pulse)
            ax.set_title('mean pulse shape')
            plt.show()


def main():
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
