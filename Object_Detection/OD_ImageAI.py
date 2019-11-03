from imageai.Detection.Custom import DetectionModelTrainer

obj = 'hololens'
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="C:\\Projects\\{}".format(obj))
trainer.setTrainConfig(object_names_array=[obj], batch_size=4,
                       num_experiments=10, train_from_pretrained_model="C:\\Projects\\pretrained-yolov3.h5")
trainer.trainModel()