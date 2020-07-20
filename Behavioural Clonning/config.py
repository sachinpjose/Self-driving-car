"""
Behavioural Clonning configuration and hyperparameters
"""


CONFIG = {

    'input_width': 200,
    'input_height': 66,
    'input_channels': 3,
    'num_bins' : 25,
    'samples_per_bin' : 400,
    'data_dir' : r'C:\Users\Lenovo\Workspace\Machine Learning\Computer vision\Self driving car\Behavioural clonning\track',
    'adam_lr': 1e-3,
    'steps_per_epoch': 300,
    'epochs': 10,
    'validation_steps' : 200,
    'save_model' : 'drive_model.h5',
    'load_model' : r'C:/Users/Lenovo/Workspace/Machine Learning/Computer vision/Self driving car/Behavioural clonning/pretrained/model.h5',
    'speed_limit' : 10,
    'batchsize': 512,
    'delta_correction': 0.25,
    'augmentation_steer_sigma': 0.2,
    'augmentation_value_min': 0.2,
    'augmentation_value_max': 1.5,
}

