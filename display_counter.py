import tkinter as tk
import time
import os

def read_count():
    try:
        with open("count.txt", "r") as f:
            return f.read().strip()
    except Exception:
        return "0"

def update_label():
    count = read_count()
    label.config(text=count)
    root.after(200, update_label)  # update every 200 ms

root = tk.Tk()
root.title("Object Counter")
root.geometry("600x600")
root.configure(bg="white")

title_label = tk.Label(root, text="Antal telefoner fundet", font=("Times new roman", 40), fg="black", bg="white")
title_label.pack(anchor="n")  # Top of window, no extra space

label = tk.Label(root, text="0", font=("Times new roman", 160), fg="black", bg="white")
label.pack(pady=(0, 0), expand=True)  # Remove extra space above/below

update_label()
root.mainloop()
