from keras.preprocessing import image
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import cv2

json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model1.h5') 
print("Model Loaded Successfully")

cam = cv2.VideoCapture(0)

while(True):
    frame = cam.read()[1]
    img = cv2.resize(frame, (128, 128))
    
    test_image = image.img_to_array(img)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    arr = np.array(result[0])
    # print(arr)
    maxx = np.amax(arr)
    max_prob = arr.argmax(axis=0)
    max_prob = max_prob + 1
    classes=["Background", "Diseased", "Healthy"]
    result = classes[max_prob - 1]    
    print(result)

    cv2.putText(frame, result, (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Leaf",frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
