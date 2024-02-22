import pyttsx3
import torch
import numpy as np
from ultralytics import YOLO


def text_to_sound(text):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', 125)
    
    engine.say(text)
    
    engine.runAndWait()



def detect_classes(results, model):
    boxes = results[0].boxes.cpu().numpy()
    xyxys = []
    confidences = []
    class_ids = []
    
    xyxys.append(boxes.xyxy)
    confidences.append(boxes.conf)    
    class_ids.append(boxes.cls)    
    
    if class_ids[0] is not None and len(class_ids[0]) > 0:
       detected_class_index = int(class_ids[0][0])
       detected_class_name = model.names.get(detected_class_index, f'Unknown-{detected_class_index}')
       if detected_class_name == 'K' and class_ids[0] is not None and len(class_ids[0]) > 1:
            detected_class_index = int(class_ids[0][1])
            detected_class_name = model.names.get(detected_class_index, f'Unknown-{detected_class_index}')
            print('Letra K ignorada')
       if detected_class_name == 'U' and class_ids[0] is not None and len(class_ids[0]) > 1:
            detected_class_index = int(class_ids[0][1])
            detected_class_name = model.names.get(detected_class_index, f'Unknown-{detected_class_index}')
            print('Letra U ignorada')
       if detected_class_name == 'P' and class_ids[0] is not None and len(class_ids[0]) > 1:
            detected_class_index = int(class_ids[0][1])
            detected_class_name = model.names.get(detected_class_index, f'Unknown-{detected_class_index}')
            print('Letra P ignorada')
            
       print('Clase detectada:', detected_class_name)
       return detected_class_name
    return None





# names: {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
# 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
# 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
# 24: 'Y', 25: 'Z', 26: 'additional', 27: 'alcohol', 28: 'allergy',
# 29: 'bacon', 30: 'bag', 31: 'barbecue', 32: 'bill', 33: 'biscuit', 
# 34: 'bitter', 35: 'bread', 36: 'burger', 37: 'bye', 38: 'cake',
# 39: 'cash', 40: 'cheese', 41: 'chicken', 42: 'coke', 43: 'cold',
# 44: 'cost', 45: 'coupon', 46: 'credit card', 47: 'cup', 48: 'dessert',
# 49: 'drink', 50: 'drive', 51: 'eat', 52: 'eggs', 53: 'enjoy', 54: 'fork',
# 55: 'french fries', 56: 'fresh', 57: 'hello', 58: 'hot', 59: 'icecream',
# 60: 'ingredients', 61: 'juicy', 62: 'ketchup', 63: 'lactose', 64: 'lettuce',
# 65: 'lid', 66: 'manager', 67: 'menu', 68: 'milk', 69: 'mustard', 70: 'napkin',
# 71: 'no', 72: 'order', 73: 'pepper', 74: 'pickle', 75: 'pizza', 76: 'please',
# 77: 'ready', 78: 'receipt', 79: 'refill', 80: 'repeat', 81: 'safe', 82: 'salt',
# 83: 'sandwich', 84: 'sauce', 85: 'small', 86: 'soda', 87: 'sorry', 88: 'spicy',
# 89: 'spoon', 90: 'straw', 91: 'sugar', 92: 'sweet', 93: 'thank-you',
# 94: 'tissues', 95: 'tomato', 96: 'total', 97: 'urgent', 98: 'vegetables',
# 99: 'wait', 100: 'warm', 101: 'water', 102: 'what', 103: 'would', 104: 'yoghurt',
# 105: 'your'}