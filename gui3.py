import tkinter as tk
from tkinter import filedialog
import functions as f
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve, find_peaks
from scipy.optimize import curve_fit
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
        self.window_entry.insert(0, 1000)
        self.window_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        thres_label = tk.Label(self.master, text='threshold')
        thres_label.grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.thres_entry = tk.Entry(self.master)
        self.thres_entry.insert(0, 5)
        self.thres_entry.grid(row=0, column=3, sticky="w", padx=5, pady=5)

        thres_units = ["stds", "resp"]
        self.thres_unit = tk.StringVar(self.master)
        self.thres_unit.set(thres_units[0])  # Set initial value
        self.thres_unit_menu = tk.OptionMenu(self.master, self.thres_unit, *thres_units)
        self.thres_unit_menu.grid(row=0, column=4, sticky="w", padx=5, pady=5)


        smooth_label = tk.Label(self.master, text='lifetime \n for smoothing [us]')
        smooth_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.smooth_entry = tk.Entry(self.master)
        self.smooth_entry.insert(0, 'optional')
        self.smooth_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        windowtypes = ["box", "exp"]
        self.windowtype = tk.StringVar(self.master)
        self.windowtype.set(windowtypes[0])  # Set initial value
        windowtype_label = tk.Label(self.master, text='window type')
        windowtype_label.grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.windowtype_menu = tk.OptionMenu(self.master, self.windowtype, *windowtypes)
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
        if path[-4:] != '.bin':
            raise Exception('Please select a .bin file')
        info_path = path[:-4] + '_info.dat'
        try:
            info = f.get_info(info_path)
            sf = int(info[-2])
        except:
            sf = f.ensure_type(input('No info file found at: %s. Please input the sampling frequency as integer: \n' % info_path), int)
        dt = int(1 / sf * 1e6)
        tqp = f.ensure_type(self.smooth_entry.get(), int, orNoneType=True)
        window = f.ensure_type(self.windowtype.get(), str)
        if tqp:
            filter = f.get_window(window, int(tqp / dt))
            sw = len(filter)
        thres = f.ensure_type(self.thres_entry.get(), (float, int))
        thres_unit = f.ensure_type(self.thres_unit.get(), str)
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
                _, theta, _ = f.concat_vis(files[:nr_files])
                name = files[0].split('/')[-1]
            else:
                self.dark_dir = path
                _, theta = f.bin2mat(path)
            
            nr_points = len(theta)
            t = np.arange(nr_points) / sf
            plot_idx = int(nr_points)
            max = np.ceil(np.amax(theta[:plot_idx]))
            max = (np.amax(theta[:plot_idx]) // .5 + 1) *.5
            min = (np.amin(theta[:plot_idx]) // -.1 + 1) *-.1
            
            fig, axes = plt.subplot_mosaic('aabc', figsize=(12, 3), constrained_layout=True, sharey=True, num=str(path))
            ax = axes['a']
            ax.set_ylim([min, max])
            ax.set_xlim([0, t[plot_idx-1]])
            ax.set_xlabel('')
            ax.set_ylabel('Response')
            ax.set_xlabel('time [$s$]')
            ax.set_title(name)
            if tqp:
                signal = convolve(theta, filter, mode='valid')
                lbl='smoothed'
                clr = 'tab:orange'
                ax.plot(t[:plot_idx-sw], signal[:plot_idx-sw], lw=.5, label=lbl, zorder=1, c=clr)
            else:
                signal = theta
                lbl = 'unsmoothed'
                clr = 'tab:blue'
                ax.plot(t[:plot_idx], signal[:plot_idx], lw=.5, label=lbl, zorder=0)
            if thres:
                if thres_unit == 'stds':
                    neg_signal = signal[signal<=0]
                    std = np.std(np.hstack((neg_signal, np.absolute(neg_signal))))
                    std = np.round(std, decimals=3)
                    mph = thres * std
                    mph = thres * np.std(signal)
                elif thres_unit == 'resp':
                    mph = thres

                locs, props = find_peaks(signal, height=mph, prominence=mph/2) 
                heights = props['peak_heights']
                nr_peaks = len(heights)
                peak_rate = nr_peaks / (nr_points / sf)
                ax.axhline(mph, color='tab:red', lw=1, label='mph=%.3f' % mph, zorder=2)
            ax.legend(loc='upper right')
            ax = axes['b']
            if thres and pw:
                ax.axhline(mph, color='tab:red', lw=1, zorder=2)
                ax.axhline(0, color='k', lw=0.5, zorder=0)
                ax.hist(heights, 'auto', facecolor='tab:blue', label='$N_{ph}$=%.f cps' % peak_rate, orientation=u'horizontal')
                ax.set_xlabel('Counts')
                ax.legend()
                ax.set_title('Heights, %d files' % (nr_files))
                
                pw = int(pw / dt)
                pulses = []
                offset = int(np.ceil(50 / dt))
                too_close = 0
                single_pulses = np.zeros(nr_peaks, dtype=bool)
                for i, loc in enumerate(locs):
                    too_close = 0
                    if  i < nr_peaks - 1 and i > 0: 
                        if (loc + pw >= locs[i+1] or loc - pw - offset <= locs[i-1] or loc + pw >= nr_points or loc - offset < 0):
                            too_close = 1
                    elif i == 0:
                        if nr_peaks > 1:
                            if (loc + pw >= locs[i+1] or loc + pw >= nr_points or loc - offset < 0):
                                too_close = 1
                        else:
                            if (loc + pw >= nr_points or loc - offset < 0):
                                too_close = 1
                    elif i == nr_peaks - 1: 
                        if (loc - pw - offset <= locs[i-1] or loc + pw >= nr_points or loc - offset < 0):
                            too_close = 1 
                    if not too_close:                    
                        single_pulses[i] = 1
                        pulse = theta[loc-offset:loc+pw]
                        pulses.append(pulse)
                nr_too_close = np.sum(~single_pulses)
                axes['a'].scatter(locs[single_pulses] / sf, heights[single_pulses], marker='v', c='None', edgecolors='tab:green', lw=1, label='peaks', zorder=3)
                if nr_too_close:
                    print('%d peaks too close' % nr_too_close)
                    axes['a'].scatter(locs[~single_pulses] / sf, heights[~single_pulses], marker='v', c='None', edgecolors='tab:red', lw=1, label='too close', zorder=3)
                # axes['a'].legend(loc='upper right')
                ax.hist(heights[single_pulses], 'auto', facecolor='tab:green', label='singles', orientation=u'horizontal')
                pulses = np.array(pulses)
                pulses = pulses.reshape((-1, pw+offset))
                nr_sel_pulses = pulses.shape[0]
                if nr_sel_pulses:
                    mean_pulse = np.mean(pulses, axis=0)
                    fit_l = int(0.1*pw)
                    fit_r = int(0.5*pw)
                    fit_x = np.arange(fit_l, fit_r)
                    fit_y = mean_pulse[fit_l+offset:fit_r+offset]
                    popt, _ = curve_fit(f.exp_decay, (fit_x-fit_l)*dt, fit_y)
                    tau = 1 / popt[1]
                    fit_y = f.exp_decay((fit_x-fit_l)*dt, *popt)
                    ax = axes['c']
                    ax.axhline(0, color='k', lw=0.5, zorder=0)
                    t_pulse = np.linspace(-offset*dt, pw*dt, len(mean_pulse))
                    ax.plot(t_pulse, mean_pulse)
                    ax.plot(fit_x*dt, fit_y, ls='--', c='tab:orange', label='$\\tau_{qp}$=%d $\mu$s' % tau)
                    ax.set_xlabel('time [$\mu s$]')
                    ax.set_xlim([t_pulse[0], t_pulse[-1]])
                    inset = ax.inset_axes([.5, .5, .45, .45])
                    inset.semilogy(t_pulse, mean_pulse)
                    inset.semilogy(fit_x*dt, fit_y, ls='--', c='tab:orange')
                    inset.set_xlim([t_pulse[0], t_pulse[-1]])
                    ax.set_title('mean pulse shape')
                    ax.legend(loc='lower right')
            plt.show()


def main():
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
