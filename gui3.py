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

        window_label = tk.Label(self.master, text='pulse window [us]')
        window_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.window_entry = tk.Entry(self.master)
        self.window_entry.insert(0, 200)
        self.window_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        thres_label = tk.Label(self.master, text='threshold [# \u03C3]')
        thres_label.grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.thres_entry = tk.Entry(self.master)
        self.thres_entry.insert(0, 3)
        self.thres_entry.grid(row=0, column=3, sticky="w", padx=5, pady=5)


        smooth_label = tk.Label(self.master, text='lifetime \n for smoothing [us]')
        smooth_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.smooth_entry = tk.Entry(self.master)
        self.smooth_entry.insert(0, 'optional')
        self.smooth_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        windowtypes = ["box", "exp"]
        self.selected_option = tk.StringVar(self.master)
        self.selected_option.set(windowtypes[0])  # Set initial value
        windowtype_label = tk.Label(self.master, text='window type')
        windowtype_label.grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.windowtype_menu = tk.OptionMenu(self.master, self.selected_option, *windowtypes)
        self.windowtype_menu.grid(row=1, column=3, sticky="w", padx=5, pady=5)


        nrfiles_label = tk.Label(self.master, text='# of files')
        nrfiles_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.nrfiles_entry = tk.Entry(self.master)
        self.nrfiles_entry.insert(0, 1)
        self.nrfiles_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        path_label = tk.Label(self.master, text='analyse')
        path_label.grid(row=2, column=2, sticky="w", padx=5, pady=5)
        path_button = tk.Button(self.master, text="Select file", command=self.select_directory)
        path_button.grid(row=2, column=3, sticky="w", padx=5, pady=5)

    def select_directory(self):
        path = filedialog.askopenfilename()
        info_path = path[:-4] + '_info.dat'
        try:
            info = f.get_info(info_path)
            sf = int(info['fs'])
        except:
            print('No info file found, taking sf=1e6')
            sf = int(1e6)
        dt = 1 / sf * 1e6
        tqp = f.ensure_type(self.smooth_entry.get(), int, orNoneType=True)
        window = f.ensure_type(self.selected_option.get(), str)
        if tqp:
            sw = int(tqp / dt)
            filter = f.get_window(window, tqp)
            sw = len(filter)
        nr_stds = f.ensure_type(self.thres_entry.get(), int)
        nr_files = f.ensure_type(self.nrfiles_entry.get(), int)
        pw = f.ensure_type(self.window_entry.get(), int)
        if path:
            name = path.split('/')[-1]
            if (nr_files > 1) and ('vis' in name):
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
            t = np.arange(nr_points) / sf
            # plot_idx = int(nr_points / nr_files)
            plot_idx = int(nr_points)
            max = np.ceil(np.amax(X[:plot_idx]))
            min = (np.amin(X[:plot_idx]) // -.5 + 1) *-.5
            
            fig, axes = plt.subplot_mosaic('aabc', figsize=(12, 3), constrained_layout=True, sharey=True)
            ax = axes['a']
            ax.plot(t[:plot_idx], X[:plot_idx], lw=.5, label='Im(z)', zorder=0)
            ax.set_xlim([0, t[plot_idx-1]])
            ax.set_ylim([min, max])
            ax.set_xlabel('')
            ax.set_ylabel('Response')
            ax.set_xlabel('time [$s$]')
            ax.set_title(name)
            if tqp:
                Xsmooth = convolve(X, filter, mode='valid')
                ax.plot(t[:plot_idx-sw], Xsmooth[:plot_idx-sw], lw=.5, label='smoothed Im(z)', zorder=1)
            if nr_stds:
                if tqp:
                    mph = nr_stds * np.std(Xsmooth)
                    locs, props = find_peaks(Xsmooth, height=mph, prominence=mph) 
                    heights = X[locs]
                else:
                    mph = nr_stds * np.std(X)
                    locs, props = find_peaks(X, height=mph, prominence=mph) 
                    heights = props['peak_heights']
                nr_peaks = len(heights)
                peak_rate = nr_peaks / (nr_points / sf)
                plot_locs = locs < plot_idx
                ax.scatter(locs[plot_locs] / sf, heights[plot_locs], marker='v', c='None', edgecolors='tab:green', lw=1, label='peaks', zorder=3)
                ax.axhline(mph, color='tab:red', lw=1, label='threshold', zorder=2)
            ax.legend(loc='upper right')
            ax = axes['b']
            ax.hist(heights, 'auto', facecolor='tab:green', label='Nph=%.f cps' % peak_rate, orientation=u'horizontal')
            ax.set_xlabel('Counts')
            ax.legend()
            ax.set_title('Heights vis%d-%d' % (0, nr_files))
            if pw:
                pw = int(pw / dt)
                pulses = []
                offset = int(np.ceil(int(25 / dt)))
                too_close = 0
                for i, loc in enumerate(locs):
                    if  i < nr_peaks-1 and loc + pw + offset >= locs[i+1] and loc + pw <= nr_points and loc - offset >= 0:
                        too_close += 1
                    else:
                        pulses.append(X[loc-offset:loc+pw])
                if too_close:
                    print('%d peaks too close' % too_close)
                
                pulses = np.array(pulses).reshape((-1, pw+offset))
                if nr_peaks:
                    mean_pulse = np.mean(pulses, axis=0)
                    ax = axes['c']
                    t_pulse = np.linspace(-offset*dt, pw*dt, len(mean_pulse))
                    ax.plot(t_pulse, mean_pulse)
                    ax.set_xlabel('time [$\mu s$]')
                    ax.set_xlim([t_pulse[0], t_pulse[-1]])
                    inset = ax.inset_axes([.5, .5, .45, .45])
                    inset.semilogy(t_pulse, mean_pulse)
                    inset.set_xlim([t_pulse[0], t_pulse[-1]])
                    ax.set_title('mean pulse shape')
            plt.show()


def main():
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
