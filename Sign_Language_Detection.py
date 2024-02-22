print('Importando librerias...')
import cv2
from ultralytics import YOLO
from utils import text_to_sound, detect_classes
from sense_hat import SenseHat
print('Librerias importadas.')

print('Inicializando sense_hat...')
sense = SenseHat()
sense.clear()
print('Sense_hat inicializado.')

print('Inicializando YOLOv8...')
model = YOLO('SignLanguageModel003.pt')  # initialize model
print('YOLOv8 inicializado.')

win_name = 'Detection'
video_reader = cv2.VideoCapture(0)
text = ''

while True:
    _, image = video_reader.read()
    cv2.imshow(win_name, image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('w'):
        results = model(image, save=False, conf=0.1, show=True)      
        classes = detect_classes(results, model)
        if classes is not None:
            text += classes
            sense.show_message(text, scroll_speed=0.2, text_colour=[0, 255, 0])
        print('Texto: ', text)
        
    if key == ord('d'):
        text = text[:-1]  
        print('Texto: ', text)
        sense.show_message(text, scroll_speed=0.2, text_colour=[255, 0, 0])
        
    if key == ord('s'):
        text_to_sound(text)
        sense.show_message(text, scroll_speed=0.2, text_colour=[255, 255, 255])
        text = ''     
        
    if key == ord('q'):
        sense.show_message('Adios! :(', scroll_speed=0.2, text_colour=[255, 255, 255])
        break
  
cv2.destroyAllWindows()
video_reader.release()





