import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import sys
from unicodedata import name

# import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np
import pickle
import time
# import matplotlib.pyplot as plt

def test_tflite(saved_info, tflite_model='bittle_frozen_axis1'):
    '''Use tflite to make predictions and compare predictions to stablebaselines output'''

    #TFlite model path
    model_save_file = tflite_model + ".tflite"

    #Load tflite model
    interpreter = tflite.Interpreter(model_path=model_save_file, experimental_delegates=None)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(saved_info['obs'])):
        #Read Observations
        obs = saved_info['obs'][i]

        input_data = obs.reshape(1, -1)
        input_data = np.array(input_data, dtype=np.float32)
        # print("Observation",input_data)
        interpreter.set_tensor(input_details[0]['index'],input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data).flatten()
        # print("TFLite Output:",output_data)
        # print("SB Saved Output",saved_info['actions'][i])
        # print('\n')

        #Send action:

        #Receive Observations:

def get_saved_info(file_name):
    #Get verification saved obs and actions
    with open(file_name+'.pickle','rb') as handle:
        saved_info = pickle.load(handle)
    return saved_info

if __name__ == "__main__":
    #Frozen TFLite Model Directory
    tflite_model = 'output/bittle_frozen_axis1'

    trials = 5
    trial_process_times_lst = []

    #Run 5 Different Saved Info Files 5 Times
    for i in range(trials):
        for i in range(trials):
            #Get saved info to pass into model
            file_number = i + 1
            saved_info = get_saved_info('saved_info_2ep_1000steps' + str(file_number))

            #Start Process timer
            start_process_time = time.process_time()
            #Run model for the saved observations
            test_tflite(saved_info,tflite_model)
            #Calculate Process Time
            trial_process_time = time.process_time()-start_process_time
            
            #Save Times
            trial_process_times_lst.append(trial_process_time)
    
    #Calculate Average Time
    average_time = 0
    for i in range(len(trial_process_times_lst)):
        average_time+=trial_process_times_lst[i]
    average_time/=(trials*trials)

    # #Plot
    # plt.scatter(list(range(len(trial_process_times_lst))) ,trial_process_times_lst)
    # plt.show()
    # print('Average Time:',average_time)