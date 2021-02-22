* directory

Appearance  - script for cnn
datasets    - dataset for tracking
demo        - script of tracking 
keras_yolo3 - yolo v3 detector
model       - script of multi vehicle tracking algorithm model
tools       - script of training and tracking
utils       - script of utilities

[Appearance]
* Appearance - datasets
             - logs
             - model
             - tools
             - utils

1) generate dataset to train CNN(appearance extractor) => check [ Appearcne -> datasets -> kitti_tracking -> make_label_file.py]

2) to train CNN => check [ Appearance -> tools -> training_with_online_mining.py 
                                      -> utils -> data_server_online_mining.py ]
   ==> you can check option to run in the both scripts  
   ==> to train, you need to run server script to provide training dataset

[Whole model]

1) generate dataset to train model => check [ dataset -> kitti_tracking -> make_label_file.py ]

2) to train model => check [ tools -> kl_trainig.py
                             utils -> data_server.py]

   ==> you can check option to run in the both scripts  
   ==> to train, you need to run server script to proivde training dataset

3) to tracking with video => check [tool -> tracking]

