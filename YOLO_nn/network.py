from ultralytics import YOLO, settings
import os
import matplotlib.pyplot as plt
#
#
CURRENT_DIR = os.getcwd()
BEST_MODEL_PATH = os.path.join(CURRENT_DIR, 'runs\detect\\best_train\weights\\best.pt')
model = YOLO(BEST_MODEL_PATH)

print(model.info())
results = model.predict(source=CURRENT_DIR + '/test_images')

for result in results:
    result.show()
    for box in result.boxes:
        xmin, ymin, xmax, ymax = box.xyxy[0]
        print(f"""Класс: {model.names[int(box.cls)]}
Координаты рамки
Левая сторона: {xmin:.2f}
Правая сторона: {xmax:.2f}
Верхняя сторона: {ymin:.2f}
Нижняя сторона : {ymax:.2f}
Уверенность: {float(box.conf):.2f}""")

