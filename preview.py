"""Side-by-side window of the intermediate images, for checking a run by eye."""

import numpy as np

PANEL_SIZE = 400


def show_preview(panels: list[tuple[str, np.ndarray]]) -> None:
    """Blocks until the window is closed. Tk is imported here so runs with
    --no-preview work on Pythons built without it."""
    import tkinter as tk

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("img2depth preview")
    photos = []  # Tk drops images that are not referenced from Python.
    for title, image in panels:
        frame = tk.Frame(root)
        frame.pack(side=tk.LEFT, padx=5)
        tk.Label(frame, text=title).pack()
        picture = Image.fromarray(image)
        picture.thumbnail((PANEL_SIZE, PANEL_SIZE))
        photos.append(ImageTk.PhotoImage(picture))
        tk.Label(frame, image=photos[-1]).pack()
    root.mainloop()
