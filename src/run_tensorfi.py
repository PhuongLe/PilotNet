import sys
sys.path.insert(0, "/home/phoebe/capstone/PilotNet")

import tensorflow as tf
import scipy.misc
import cv2
from subprocess import call
from nets.pilotNet import PilotNet
import logging

import time
import TensorFI as ti
import datetime

FLAGS = tf.app.flags.FLAGS
numOfInjection = 100
sdcMax = 5

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
overalReport = open("reports/overalReport.csv", "w")
overalReport.write("image,golden_value,average_difference,sdc_percentage\n")
detailsReport = open("reports/detailsReport.csv", "w")
detailsReport.write("image,golden_value")

for idx in range(numOfInjection):
    detailsReport.write(",error_" + str(idx))

detailsReport.write("\n")

# initialize TensorFI 
fi = ti.TensorFI(sess, logLevel = logging.DEBUG, name = "PilotNet", disableInjections=True)
# inputs to be injected
#index = [20, 486, 992, 1398, 4429, 5259, 5868, 6350, 6650, 7771]
index = [20,40,60,80]
#index = [20]
averageSDC = 0
for i in index:
    full_image = scipy.misc.imread(FLAGS.dataset_dir + "/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    overalReport.write(str(i) + ",")
    detailsReport.write(str(i) + ",")

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
 
    totalFI = 0
    sdcCount = 0  
    errorAverage = 0
    
    overalReport.write(str(golden) + ",")
    detailsReport.write(str(golden) + ",")
    print("golden steering = " + str(golden))

    # keep doing FI on each injection point until the end
    for i in range(numOfInjection): 
        steering = sess.run(
                    model.steering,
                    feed_dict={
                        model.image_input: [image],
                        model.keep_prob: 1.0
                    }
                )

        if steering is None:
            degrees = 10000
            print("predicted steering = None")
        else:
            degrees = steering[0][0] * 180.0 / scipy.pi
            print("predicted steering = " + str(degrees))

        totalFI += 1

        difference = round(abs(degrees - golden),2)
        if difference > sdcMax:
            sdcCount +=1            
        
        errorAverage+=degrees

        # we store the value of the deviation, so that we can use different thresholds to parse the results
        detailsReport.write(str(difference) + ",")

    errorAverage = float(errorAverage)/float(numOfInjection)
    sdcPercentage = 100 * float(sdcCount)/float(totalFI)
    overalReport.write(str(errorAverage) + ",")
    overalReport.write(str(sdcPercentage) + ",")
    averageSDC = averageSDC + sdcPercentage    
    #print(i, totalFI, sdcCount)

    overalReport.write("\n")
    detailsReport.write("\n")

averageSDC = averageSDC/4
overalReport.write("overall average SDC = " + str(averageSDC))
print("overall average SDC = " + str(averageSDC))
#def diffFunc()