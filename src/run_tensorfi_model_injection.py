import sys
sys.path.insert(0, "/home/phoebe/capstone/PilotNet")

import tensorflow as tf
import scipy.misc
#import model
import cv2
from subprocess import call
from nets.pilotNet import PilotNet
import logging

#import driving_data
import time
import TensorFI as ti
import datetime

FLAGS = tf.app.flags.FLAGS

"""model from nvidia's training"""
tf.app.flags.DEFINE_string(
    'model_file', './data/models/nvidia/model.ckpt',
    """Path to the model parameter file.""")

tf.app.flags.DEFINE_string(
    'dataset_dir', './data/datasets/driving_dataset',
    """Directory that stores input recored front view images.""")

model = PilotNet()

sess = tf.Session()
saver = tf.train.Saver()
#### Important: make sure you've trained the model, refer to train.py
saver.restore(sess, FLAGS.model_file)


# save each FI result
resFile = open("eachFIres.csv", "w")


# initialize TensorFI 
fi = ti.TensorFI(sess, logLevel = logging.DEBUG, name = "PilotNet", disableInjections=True)
# inputs to be injected
#index = [20, 486, 992, 1398, 4429, 5259, 5868, 6350, 6650, 7771]
index = [20]
#while(cv2.waitKey(10) != ord('q')):


for i in index:
    full_image = scipy.misc.imread(FLAGS.dataset_dir + "/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    resFile.write(str(i) + ",")

    # we first need to obtain the steering angle in the fault-free run
    fi.turnOffInjections()

    steering = sess.run(
                    model.steering,
                    feed_dict={
                        model.image_input: [image],
                        model.keep_prob: 1.0
                    }
                )


    degrees = steering[0][0] * 180.0 / scipy.pi

    golden = degrees 

    # perform FI
    fi.turnOnInjections() 
 
    totalFI = 0.
    sdcCount = 0  
    numOfInjection = 1
    
    resFile.write("golden = " + str(golden) + ",")

    #tf.reset_default_graph()
    # keep doing FI on each injection point until the end
    for i in range(numOfInjection): 

        #sess.run(tf.compat.v1.global_variables_initializer())
        #writer = tf.compat.v1.summary.FileWriter('./figraphs', graph=sess.graph)
        steering = sess.run(
                    model.steering,
                    feed_dict={
                        model.image_input: [image],
                        model.keep_prob: 1.0
                    }
                )
        
        print("Input tensor = " + str(model.steering))
        print("golden steering = " + str(golden))
        if steering is None:
            print("predicted steering = None")
        else:
            degrees = steering[0][0] * 180.0 / scipy.pi
            print("predicted steering = " + str(degrees))

        totalFI += 1
        # we store the value of the deviation, so that we can use different thresholds to parse the results
        resFile.write(str(abs(degrees - golden)) + ",")
        print(i, totalFI)

    resFile.write("\n")
