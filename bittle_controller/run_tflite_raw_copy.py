
import tflite_runtime.interpreter as tflite
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import argparse

from serialMaster.policy2serial import *

import action_filter

STEPS = 100
INITIAL_POSE = [.52]*8

def verify_tflite(interpreter, saved_info, show_output=False):
    '''Use tflite to make predictions and compare predictions to stablebaselines output
    
    interpreter: Loaded frozen tflite model
    saved_info: saved obs and actions from SB model which will be used to compare to tflite output
    show_output: plot the results
    '''
    #Prepare tflite model
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(saved_info['obs'])):
        #Read Observations
        obs = saved_info['obs'][i]

        input_data = obs.reshape(1, -1)
        input_data = np.array(input_data, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'],input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data).flatten()

        #Display output of TFLite and SB Models
        if show_output:
            print("TFLite Output:", output_data)
            print("SB Saved Output", saved_info['actions'][i])
            print('\n')


def model_processing_time(plot_time=False, trials=5):
    '''#Run 5 Different Saved Info Files 5 Times
        Each info file consists of two episodes with 500 steps each'''
    
    trial_process_times_lst = []

    #Run 5 Different Saved Info Files 5 Times
    for i in range(trials):
        for i in range(trials):
            #Get saved info to pass into model
            file_number = i + 1
            saved_info = get_saved_info('data/saved_info_2ep_1000steps' + str(file_number))

            #Load the model
            tflite_interpreter = load_model(args.tflite_model)

            #Start Process timer
            start_process_time = time.process_time()
            #Run model for the saved observations
            verify_tflite(tflite_interpreter,saved_info)
            #Calculate Process Time
            trial_process_time = time.process_time()-start_process_time
            
            #Save Times
            trial_process_times_lst.append(trial_process_time)
    
    #Calculate Average Time
    average_time = 0
    for i in range(len(trial_process_times_lst)):
        average_time+=trial_process_times_lst[i]
    average_time/=(trials*trials)
    print('Average Time:',average_time)

    #Plot
    if plot_time:
        plt.scatter(list(range(len(trial_process_times_lst))) ,trial_process_times_lst)
        plt.title('TFLite Time to process 1000 Steps for 25 Trials')
        plt.xlabel('Trial')
        plt.ylabel('Time (s)')
        plt.savefig('captures/tflite_time_test.png')


def deploy_on_bittle(interpreter):
    '''Deploy model on real life bittle and receive feedback
    
    interpreter: Loaded frozen tflite model
    '''
    global step_counter
    angle_analysis = False

    #Prepare tflite model
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #initialize commands
    initializeCommands()

    #initialize pose
    step_real_bittle(INITIAL_POSE, True)

    time.sleep(.5)

    #get initial obs
    _, imu_sensor = getBittleIMUSensorInfo()

    #initialize last action queue
    last_action_queue = [0]*24
    #initialize joint angle queue
    joint_angle_queue = INITIAL_POSE + [0]*16

    #IMU: 0-11, Last Action: 12-35, Motor Angle: 36-59, Target: 60 - 119
    obs = np.concatenate((imu_sensor,last_action_queue,joint_angle_queue))

    delta_lst = []
    previous_saved_action = [0]*8

    with open('output/saved_info_1ep_model7.pickle','rb') as handle:
        saved_info = pickle.load(handle)

    for i in range(STEPS):
        obs = saved_info['obs'][i]

        input_data = obs.reshape(1, -1)
        input_data = np.array(input_data, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'],input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data).flatten()
        output_data = output_data[:8]

        #Send action:
        processed_action = step_real_bittle(output_data,True)
        print('Processed',processed_action)
        print('Output',output_data)
        output_data = processed_action

        #Receive Observations:
        _, imu_sensor = getBittleIMUSensorInfo()

        #Update last action queue by appending to top and removing last 8
        last_action_queue = np.concatenate((output_data,last_action_queue))
        last_action_queue = last_action_queue[:24]
        #Update the joint angle queue
        joint_angle_queue = np.concatenate((output_data,joint_angle_queue))
        joint_angle_queue = joint_angle_queue[:24]
        
        #Create full observation with 60 dim
        obs = np.concatenate((imu_sensor, last_action_queue, joint_angle_queue))

        time.sleep(.5)

        if angle_analysis:
            for joint, angle in enumerate(output_data):
                delta_lst.append(abs(np.degrees(angle)-previous_saved_action[joint]))
                previous_saved_action = output_data

        step_counter+=1
        
    if angle_analysis:
        plt.hist(delta_lst, bins=30)
        plt.show()
        print(f'The average angle change is: {sum(delta_lst)/len(delta_lst)}')


def step_real_bittle(action, apply_changes):
    if apply_changes:
        proc_action = step(np.array(action))

    # change actions to degrees
    action = np.degrees(proc_action)

    # set all joint angles simultaneously
    task = ['i',[9,action[0],13,action[1],8,action[2],12,action[3],10, action[4],14,action[5],11, action[6],15, action[7]],0]
    sendCommand(task)

    return proc_action


_action_filter = None
step_counter = 0

######## REMOVE -----------------------------------------
def step(action):
    filtered_action = _FilterAction(action)

    return filtered_action


def _FilterAction(action):
        # initialize the filter history, since resetting the filter will fill
    # the history with zeros and this can cause sudden movements at the start
    # of each episode
    global _action_filter
    if step_counter == 0:
        default_action = INITIAL_POSE 
        _action_filter = _BuildActionFilter()
        _action_filter.reset()
        _action_filter.init_history(default_action)

    filtered_action = _action_filter.filter(action)
    return filtered_action

def _BuildActionFilter():
    time_step = .001
    action_repeat = 33
    num_joints = 8
    sampling_rate = 1 / (time_step * action_repeat) #30
    a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate,
                                                num_joints=num_joints)
    return a_filter

#################----------------------------------------------

def manually_control_bittle():
    '''Manually send commands to bittle'''
 
    #initialize commands
    initializeCommands()

    # while True:
    #     action = [52]*8
    #     index = int(input('Which index would you like to change? (0-7)'))
    #     if index >= 0 and index < 8:
    #         joint_angle = float(input('What angle would you like to move joint to?'))
    #         action[index] = joint_angle
    #         task = ['i',[9,action[0],13,action[1],8,action[2],12, action[3], 10, action[4],14, action[5],11, action[6],15, action[7]],0]
    #         sendCommand(task)
    #         print("Command sent")
    #         time.sleep(1)
    #     else:
    #         print('Invalid index')

    joint_angle = 0
    index = 0
    action = [0]*8
    while action[index] < 30:
        action[index]+=.135
        print(action[index])
        task = ['i',[9,action[0],13,action[1],8,action[2],12, action[3], 10, action[4],14, action[5],11, action[6],15, action[7]],0]
        sendCommand(task)
        time.sleep(.5)


def load_model(tflite_model_name):
    '''Load and return a TFLite model
    
    tflite_model_name: tflite file name
    '''
    #TFlite model path
    model_save_file = tflite_model_name + ".tflite"

    #Load tflite model
    interpreter = tflite.Interpreter(model_path=model_save_file, experimental_delegates=None)

    return interpreter    

def get_saved_info(file_name):
    #Get verification saved obs and actions
    with open(file_name+'.pickle','rb') as handle:
        saved_info = pickle.load(handle)
    return saved_info

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", dest="mode", type=str, default='test')
    arg_parser.add_argument("--tflite_model", dest="tflite_model", type=str, default='output/bittle_frozen_model7')
    arg_parser.add_argument("--verification_info", dest="verification_info", type=str, default='data/saved_info_1ep')

    args = arg_parser.parse_args()

    #Frozen TFLite Model Directory
    tflite_model = args.tflite_model
    mode = args.mode

    if mode == 'time':
        #Plot the timing results 
        plot_time_answer = False
        plot_time_answer = input("Would you like to plot the time for each trial? [Y/N] ")
        if plot_time_answer.lower() == "yes" or plot_time_answer.lower() == "y":
            plot_time_answer = True

        #Time the processing time of the tflite model
        model_processing_time(plot_time_answer)

    elif mode == 'verify':
        #Get saved verification data
        saved_info = get_saved_info(args.verification_info)

        #Load the model
        tflite_interpreter = load_model(args.tflite_model)

        #Verify TFLite model output with Stable Baselines model output
        verify_tflite(tflite_interpreter, saved_info, show_output=True)

    elif mode == 'deploy':
        #Load the model
        tflite_interpreter = load_model(args.tflite_model)

        #Predict actions using TFlite model and send commands to real bittle
        deploy_on_bittle(tflite_interpreter)

    elif mode == 'manual':
        manually_control_bittle()        






        

