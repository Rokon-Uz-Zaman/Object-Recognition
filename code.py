from imageai.Detection import ObjectDetection

import os 

execution_path=os.getcwd()

detector=ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "/content/drive/MyDrive/Colab Notebooks/CV/Object detection resnet  /resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
detector.detectObjectsFromImage(input_image=os.path.join(execution_path,"/content/drive/MyDrive/Colab Notebooks/CV/Object detection resnet  /image2.jpg"), output_image_path=os.path.join(execution_path , "/content/drive/MyDrive/Colab Notebooks/CV/Object detection resnet  /output/image2.jpg"))

#print detected object and percentage probability 

for eachObject in detections:
    print(eachObject["name"]," : ",eachObject["percentage_probability"]) 
