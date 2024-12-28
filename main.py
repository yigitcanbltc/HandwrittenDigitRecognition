import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
import os

# Model dosyasının yolunu kontrol et ve yükle
model_path = 'handwritten_digit_model.keras'
if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(__file__), 'handwritten_digit_model.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

model = load_model(model_path)

class DrawingCanvas:
    def __init__(self, root):
        self.root = root
        self.root.title("El Yazısı Rakam Tanıma")
        self.root.geometry("400x600")  # Sabit pencere boyutu
        self.root.resizable(False, False)  # Pencere boyutu değiştirilemez
        
        # Ana stil ayarları
        self.style = ttk.Style()
        self.style.configure('Custom.TButton', 
                           padding=10, 
                           font=('Arial', 12))
        
        # Başlık etiketi
        self.title_label = tk.Label(root, 
                                  text="El Yazısı Rakam Tanıma", 
                                  font=('Arial', 20, 'bold'),
                                  pady=10)
        self.title_label.pack()
        
        # Açıklama etiketi
        self.info_label = tk.Label(root,
                                 text="Lütfen 0-9 arası bir rakam çizin",
                                 font=('Arial', 12),
                                 fg='#666666')
        self.info_label.pack()
        
        # Frame oluştur
        self.canvas_frame = tk.Frame(root, bd=2, relief='solid')
        self.canvas_frame.pack(pady=20)
        
        # Canvas oluştur
        self.canvas = tk.Canvas(self.canvas_frame, 
                              width=280, 
                              height=280, 
                              bg='black',
                              cursor="pencil")  # Kalem imleci
        self.canvas.pack()
        
        # Butonlar için frame
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)
        
        # Çizim için gerekli değişkenler
        self.image = Image.new('L', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        
        # Butonlar
        self.clear_button = ttk.Button(self.button_frame, 
                                     text="Temizle", 
                                     style='Custom.TButton',
                                     command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=10)
        
        self.predict_button = ttk.Button(self.button_frame, 
                                       text="Tahmin Et",
                                       style='Custom.TButton',
                                       command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=10)
        
        # Sonuç frame
        self.result_frame = tk.Frame(root, 
                                   bd=2, 
                                   relief='solid', 
                                   padx=20, 
                                   pady=10)
        self.result_frame.pack(pady=20)
        
        # Sonuç etiketi
        self.result_label = tk.Label(self.result_frame, 
                                   text="Tahmin bekleniyor...", 
                                   font=('Arial', 16))
        self.result_label.pack()
        
        # Güven oranı etiketi
        self.confidence_label = tk.Label(self.result_frame,
                                       text="", 
                                       font=('Arial', 12),
                                       fg='#666666')
        self.confidence_label.pack()
        
        # Mouse eventlerini bağla
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset_coordinates)
        
        # Kısayol tuşları
        self.root.bind('<Control-z>', lambda e: self.clear_canvas())  # Ctrl+Z ile temizle
        self.root.bind('<Return>', lambda e: self.predict())  # Enter ile tahmin et
    
    def paint(self, event):
        x, y = event.x, event.y
        r = 8  # Fırça kalınlığı
        
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                  fill='white', width=r*2,
                                  capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], 
                          fill='white', width=r*2)
        
        self.last_x = x
        self.last_y = y
    
    def reset_coordinates(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Tahmin bekleniyor...")
        self.confidence_label.config(text="")
    
    def predict(self):
        # Tahmin yap
        img_array = np.array(self.image.resize((28, 28)).convert('L'))
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        predictions = model.predict(img_array)
        
        digit = np.argmax(predictions[0])
        confidence = float(predictions[0][digit] * 100)
        
        self.result_label.config(text=f"Tahmin: {digit}")
        self.confidence_label.config(text=f"Güven: %{confidence:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingCanvas(root)
    root.mainloop()




