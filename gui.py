import tkinter as tk
from tkinter import filedialog
from classes import MKID
import functions as f
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Data Visualization GUI")

        self.data = None
        self.settings = None

        # # Load Data Button
        # self.load_data_button = tk.Button(master, text="Load Data", command=self.load_data)
        # self.load_data_button.pack()

        # # Load KID Button
        # self.load_data_button = tk.Button(master, text="Load KID", command=self.load_kid)
        # self.load_data_button.pack()

        # # Settings
        # self.settings_label = tk.Label(master, text="Settings:")
        # self.settings_label.pack()

        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="LT")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="Wavelength [nm]")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="KID #")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="Pread")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="Comment")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="Chuncksize")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 40)
        # self.setting_entry.pack()
        # # Example of setting using drop down menu
        # self.setting_option = tk.StringVar(master)
        # self.setting_option.set("Discard saturated vis files")
        # self.setting_dropdown = tk.OptionMenu(master, self.setting_option, "YES", "NO")
        # self.setting_entry.insert(0, 'YES')
        # self.setting_dropdown.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="sf")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.pack()
        # # Example of setting using entry
        # # Example of setting using drop down menu
        # self.setting_option = tk.StringVar(master)
        # self.setting_option.set("Coordinates")
        # self.setting_dropdown = tk.OptionMenu(master, self.setting_option, "circle", "smith")
        # self.setting_entry.insert(0, 'smith')
        # self.setting_dropdown.pack()
        # # Example of setting using entry
        # # Example of setting using drop down menu
        # self.setting_option = tk.StringVar(master)
        # self.setting_option.set("response")
        # self.setting_dropdown = tk.OptionMenu(master, self.setting_option, "phase", "amp")
        # self.setting_entry.insert(0, 'phase')
        # self.setting_dropdown.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="pw")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 200)
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="sw")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 'None')
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="align")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 'peak')
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="window")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 'box')
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="ssf")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 'None')
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="sstype")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 'upsample')
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="buffer")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 10)
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="mph")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 'None')
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="mpp")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 'None')
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="# noise segments")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 1000)
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="binsize")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 0.02)
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="Heights range")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 'None')
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="fitrange tqp")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, (100, 200))
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="max bandwidth")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 100000)
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="Time range 4plot")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, (0, 1))
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="# stds for filtering")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 5)
        # self.setting_entry.pack()
        # # Example of setting using entry
        # self.setting_entry_label = tk.Label(master, text="edge offset")
        # self.setting_entry_label.pack()
        # self.setting_entry = tk.Entry(master)
        # self.setting_entry.insert(0, 25)
        # self.setting_entry.pack()
        # Set default values for the settings
        self.default_settings = {
            "Setting 1": "Default Value 1",
            "Setting 2": "Default Value 2",
            "Setting 3": "Default Value 3",
            "Setting 4": "Default Value 4",
            "Setting 5": "Default Value 5",
            "Setting 6": "Default Value 6",
            "Setting 7": "Default Value 7",
            "Setting 8": "Default Value 8",
            "Setting 9": "Default Value 9",
            "Setting 10": "Default Value 10"
        }

        # Create settings
        self.create_settings()

        # # Run Button
        # self.run_button = tk.Button(master, text="Run Script", command=self.run_script)
        # self.run_button.pack()

        # # Plot Frame
        # self.plot_frame = tk.Frame(master)
        # self.plot_frame.pack()

        # # Save Button
        # self.save_button = tk.Button(master, text="Save", command=self.save_data)
        # self.save_button.pack()


    def create_settings(self):
        # Iterate through default settings and create labels and entry widgets
        row = 0
        column = 0
        for setting, default_value in self.default_settings.items():
            # Create label for setting name
            setting_label = tk.Label(self.master, text=setting)
            setting_label.grid(row=row, column=column, sticky="w", padx=5, pady=5)

            # Create entry widget for setting value
            setting_entry = tk.Entry(self.master)
            setting_entry.insert(0, default_value)
            setting_entry.grid(row=row, column=column+1, padx=5, pady=5)

            row += 1
            # After 5 settings, move to the next column
            if row % 5 == 0:
                row = 0
                column += 2


    def load_data(self):
        light_dir = filedialog.askdirectory()
        if light_dir:
            print("Directory selected:", light_dir)
        dark_dir = filedialog.askdirectory()
        if dark_dir:
            print("Directory selected:", dark_dir)

    def load_kid(self):
        kid_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if kid_path:
            kid = MKID(file_path=kid_path)


    def run_script(self):
        # Access settings and do something with the data
        setting1 = self.setting_option.get()
        setting2 = self.setting_entry.get()
        print("Setting 1:", setting1)
        print("Setting 2:", setting2)
        # Example: Plotting data using matplotlib
        if self.data is not None:
            plt.figure()
            plt.plot(self.data['x'], self.data['y'])
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Data Plot')
            self.plot_canvas = FigureCanvasTkAgg(plt.gcf(), master=self.plot_frame)
            self.plot_canvas.draw()
            self.plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def save_data(self):
        # Save data to a file
        pass
        # Example: Saving figure
        if self.data is not None:
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
