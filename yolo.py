from ultralytics import YOLO
import os 
import numpy
# Load a model
model = YOLO("yolo11n.pt")



#  Train the model
# train_results = model.train(
#     data="coco8.yaml",  # path to dataset YAML
#     epochs=100,  # number of training epochs
#     imgsz=640,  # training image size
#     device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
# )

# Evaluate model performance on the validation set
# metrics = model.val()

# Perform object detection on an image
# Extract and print bounding box data for left image

results_left = model("images/left/image~.png")
results_right = model("images/right/image.png")
# results_left[0].show()
# results_right[0].show()

left_center = 0
right_center = 0

print("Bounding boxes for the left image:")
for box in results_left[0].boxes:
    # Extract box coordinates, confidence score, and class
    coords = box.xyxy.numpy()  # Bounding box coordinates [x1, y1, x2, y2]
    confidence = box.conf.numpy()  # Confidence score
    class_id = box.cls.numpy()  # Class ID

    if class_id == 41:
        x_1 = coords[0][0]
        x_2 = coords[0][2]
        center = ((x_2 -x_1)/2) + x_1 
        print(f"Center of the left image cup: {center}")
        left_center = center

    # print(f"Coordinates: {coords}, Confidence: {confidence}, Class: {class_id}")

print("Bounding boxes for the right image:")
for box in results_right[0].boxes:
    # Extract box coordinates, confidence score, and class
    coords = box.xyxy.numpy()  # Bounding box coordinates [x1, y1, x2, y2]
    confidence = box.conf.numpy()  # Confidence score
    class_id = box.cls.numpy()  # Class ID

    if class_id == 41:
        x_1 = coords[0][0]
        x_2 = coords[0][2]
        center = ((x_2 -x_1)/2) + x_1 
        print(f"Center of the right image cup: {center} \n")
        right_center = center

# Pixel difference between the centers of both cups in left and right camera images
cup_dif = left_center - right_center
print(f"Pixel difference between images: {cup_dif} \n")

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model

# my_list = [1, 2, 3, 4]
# print(my_list[0])
# ____________

# [image data, box data, other stuf,]

# f Focal Length 26mm  10^-3 m
# d Camera Pixel sensor size: 1.22 micrometers (µm). 10^-6 m
# T Distance between cameras: 19 cm

D = cup_dif
f = 4.15
d = 1.22*10**-3
T = 190
print(f"D: {D}")
print(f"f: {f}")
print(f"d: {d}")
print(f"T: {T}")

z = (f/d) * (T/D)

print('Distance to cup: '+ str(z) +" mm")
print('Distance to cup: '+ str(z*0.0393701) +" in")