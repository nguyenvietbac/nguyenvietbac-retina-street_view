#This is respository for bouding box AI

#Usage : first install run in ai_model folder

python setup.py build_ext --inplace

#parameter setup in constant.ini

data_path = 'path to images'
result_path = 'path to save xml'
model_file =  'pre trained model name'   default  = 'resnet50_csv_69.h5'
confident_threshold = 0.5

#run tagging by 

python keras_retinanet_tool/run.py

#all command should be run in ai_model folder. suggest create .sh file to excute(include cd command to navigate to ai_model folder)

#file model should be save in ai_model/model_trained