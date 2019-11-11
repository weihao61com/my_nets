from imageai.Detection.Custom import DetectionModelTrainer

obj = 'tag'
HOME = '/home/weihao/Projects'
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="C:\\tmp\\hololens")
trainer.setTrainConfig(object_names_array=["hololens"], batch_size=4, num_experiments=20, train_from_pretrained_model="C:\\tmp\\pretrained-yolov3.h5")
#
# trainer.setTrainConfig(object_names_array=[obj], batch_size=4,
#                        num_experiments=10,
#                        train_from_pretrained_model="{}/pretrained-yolov3.h5".format(HOME))
trainer.trainModel()