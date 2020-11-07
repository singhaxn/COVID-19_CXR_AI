from fastai.vision.all import TensorImage, image2tensor
import cv2
import numpy as np
import PIL

# pytorch-extended_v2-20201027_02
class InputCropper:
    def __init__(self, crop_df, min_confidence=0.9):
        self.min_confidence = min_confidence
        self.crop_df = crop_df
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    
    def odd_window(shape, multiplier):
        height, width, _ = shape
        window_unit = min(height, width)
        window = int(window_unit * multiplier)
        window -= (1 - window % 2)
        return max(window, 7)
    
    def get_cropped_image(self, fname, threshold=True):
        img = cv2.imread(fname)
        
        x, y, width, height = self.get_crop_bounds(fname, (img.shape[1], img.shape[0]))
        img = img[y:y+height, x:x+width]
    
        if threshold:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            img_clahe = self.clahe.apply(img_gray)
            
            blur_window = InputCropper.odd_window(img.shape, 0.005)
            img_blur = cv2.medianBlur(img_gray, blur_window)
            ad_window = InputCropper.odd_window(img.shape, 0.02)
            img_ad = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ad_window, 2)
            th_otsu, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            img_stacked = np.stack([img_otsu, img_clahe, img_ad], axis=2)
            img_pil = PIL.Image.fromarray(img_stacked)
        else:
            img_pil = PIL.Image.fromarray(img)
        
        img_pil = TensorImage(image2tensor(img_pil))

        return img_pil
    
    def get_crop_bounds(self, fname, size):
        props = self.crop_df.loc[str(fname)]
        if props["confidence"] > self.min_confidence:
            x, y, width, height = props["x"], props["y"], props["width"], props["height"]
        else:
            x, y = 0, 0
            width, height = size
        
        return x, y, width, height