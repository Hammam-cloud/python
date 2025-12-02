# /gui/app.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np

from src.predict import predict_image

MODEL_PATH = 'model/model.h5'
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DrawApp:
    def __init__(self, root, width=280, height=280):
        self.root = root
        self.width = width
        self.height = height
        root.title('Alphabet Recognition - Draw a character')

        # Drawing canvas
        self.canvas = tk.Canvas(root, width=width, height=height, bg='white')
        self.canvas.pack()

        # Buttons
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(fill=tk.X)

        tk.Button(self.btn_frame, text='Predict', command=self.on_predict).pack(side=tk.LEFT)
        tk.Button(self.btn_frame, text='Clear', command=self.clear).pack(side=tk.LEFT)
        tk.Button(self.btn_frame, text='Open Image', command=self.open_image).pack(side=tk.LEFT)

        # Prediction label
        self.result_label = tk.Label(root, text='Prediction: -', font=("Arial", 16))
        self.result_label.pack()

        # PIL image for drawing
        self.image = Image.new('L', (width, height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # Bind events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', lambda e: None)

    # Draw on canvas + PIL image
    def paint(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    # Clear canvas
    def clear(self):
        self.canvas.delete('all')
        self.draw.rectangle([0, 0, self.width, self.height], fill=255)
        self.result_label.config(text='Prediction: -')

    # Load an image from disk
    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[('Image', '*.png;*.jpg;*.jpeg')])
        if not path:
            return

        img = Image.open(path).convert('L')
        img = ImageOps.fit(img, (self.width, self.height))
        self.image.paste(img)

        # Display on canvas
        self.tkimage = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tkimage)

    # Predict button: process + run model
    def on_predict(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror('Model missing', f'Model not found at {MODEL_PATH}. Run training first.')
            return

        # Resize to model size
        img = self.image.resize((28, 28))
        img = img.convert('L')

        # Convert to numpy for model
        img_np = np.array(img) / 255.0
        img_np = img_np.reshape(1, 28, 28)

        # Predict using your model
        prediction = predict_image(img_np)

        self.result_label.config(text=f'Prediction: {prediction}')


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
