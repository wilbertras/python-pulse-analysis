import tkinter as tk
from tkinter import filedialog
from classes import MKID
import functions as f
import matplotlib.pyplot as plt
from copy import copy


class GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Data Visualization GUI")

        self.data = None
        self.settings = None

        self.light_dir = ""
        self.dark_dir = ""

        self.default_settings = {
            "light_dir": "",
            "dark_dir": "",
            "LT":218,
            "wavelength":18500,
            "kid_nr": 5,
            "pread": 113,
            "chuncksize": 10,
            "max_chuncks":1,
            "comment":"",
            "sf":int(1e6),
            "coord":"smith",
            "response":"X",
            "pw": 1000,
            "sw": 250,
            "align":"peak",
            "window":"exp",
            "ssf":1,
            "sstype":"interp",
            "buffer":10,
            "mph":"",
            "mpp":"",
            "nr_noise_segments":1000,
            "binsize":0.01,
            "H_range":"",
            "fit_T":[200, 500],
            "max_bw":10000,
            "tlim":[0,1],
            "filter_std":5,
            "rise_offset": 50,
            "pulse_template":""
        }
        self.settings = copy(self.default_settings)
        self.create_settings()

    def select_directory(self, type):
        directory = filedialog.askdirectory()
        if directory:
            if type == 'light':
                self.light_dir = directory
            elif type == 'dark':
                self.dark_dir = directory
            print(str(type) + " directory selected:", directory)


    def create_settings(self):
        # Iterate through default settings and create labels and entry widgets
        row = 0
        column = 0
        nr_columns = 4
        nr_settings = len(self.default_settings.items())
        nr_rows = int(nr_settings//nr_columns)+1
        self.entries = []
        self.labels = []
        for setting, default_value in self.default_settings.items():
            # Create label for setting name
            setting_label = tk.Label(self.master, text=setting)
            setting_label.grid(row=row, column=column, sticky="w", padx=5, pady=5)
            if row == 0 and column == 0:
                setting_entry = tk.Button(self.master, text="Select Directory", command=lambda: self.select_directory('light'))
            elif row == 1 and column == 0:
                setting_entry = tk.Button(self.master, text="Select Directory", command=lambda: self.select_directory('dark'))
            else:
                # Create entry widget for setting value
                setting_entry = tk.Entry(self.master)
                setting_entry.insert(0, default_value)
                self.entries.append(setting_entry)
                self.labels.append(setting) 
            setting_entry.grid(row=row, column=column+1, padx=5, pady=5)

            row += 1
            if row % nr_rows == 0:
                row = 0
                column += 2
                # Run Button
        setting_entry = tk.Button(self.master, text="Load data", command=self.load_data)
        setting_entry.grid(row=nr_rows+1, column=0+1, padx=10, pady=10)
        setting_entry = tk.Button(self.master, text="Analyse data", command=self.overview)
        setting_entry.grid(row=nr_rows+1, column=nr_columns+1, padx=10, pady=10)
        setting_entry = tk.Button(self.master, text="Save fig", command=self.save_fig)
        setting_entry.grid(row=nr_rows+1, column=nr_columns+2, padx=10, pady=10)

    def load_data(self):
        light_dir = filedialog.askdirectory()
        if light_dir:
            print("Directory selected:", light_dir)
        dark_dir = filedialog.askdirectory()
        if dark_dir:
            print("Directory selected:", dark_dir)

    def load_data(self):
        for i, entry in enumerate(self.entries):
            self.settings[self.labels[i]] = entry.get()
        self.mkid = MKID(int(self.settings['LT']), 
                        int(self.settings['wavelength']), 
                        self.light_dir, 
                        self.dark_dir, 
                        int(self.settings['kid_nr']), 
                        int(self.settings['pread']), 
                        self.settings['comment'], 
                        int(self.settings['chuncksize']))

    def overview(self):
        self.mkid.overview(self.settings, f, max_chuncks=int(self.settings['max_chuncks']))   
        plt.show()

    def save_fig(self):
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if filename:
            plt.savefig(filename)
            print("Figure saved successfully.")


def main():
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
