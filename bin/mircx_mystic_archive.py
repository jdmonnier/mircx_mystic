import tkinter as tk
from tkinter import filedialog

def select_directories():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Default directories
    default_input_dir = "~/."
    default_output_dir = "~/DATA_ARCHIVE/"

    # Ask user to select input directory
    input_dir = filedialog.askdirectory(initialdir=default_input_dir, title="Select Input Directory")
    if not input_dir:
        input_dir = default_input_dir

    # Ask user to select output directory
    output_dir = filedialog.askdirectory(initialdir=default_output_dir, title="Select Output Directory")
    if not output_dir:
        output_dir = default_output_dir

    return input_dir, output_dir

if __name__ == "__main__":
    input_dir, output_dir = select_directories()
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")