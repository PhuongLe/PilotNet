### Nvidia Dave steering model, model structure is in model.py
### Implementation from https://github.com/SullyChen/Autopilot-TensorFlow
### Dataset available from https://github.com/SullyChen/driving-datasets



import tensorflow as tf
import scipy.misc
#import model
import cv2
from subprocess import call
from nets.pilotNet import PilotNet

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


#img = cv2.imread('steering_wheel_image.jpg',0)
#rows,cols = img.shape
#smoothed_angle = 0

# save each FI result
resFile = open("eachFIres.csv", "w")


# initialize TensorFI 
fi = ti.TensorFI(sess, logLevel = 50, name = "convolutional", disableInjections=True)

# inputs to be injected
#index = [20, 486, 992, 1398, 4429, 5259, 5868, 6350, 6650, 7771]
index = [20]
#while(cv2.waitKey(10) != ord('q')):


for i in index:
    full_image = scipy.misc.imread(FLAGS.dataset_dir + "/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    resFile.write(str(i) + ",")

    '''    
    # The commented code is for inferencing and visualization
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi 
#    call("clear")
    print(i , ".png", " Predicted steering angle: " + str(degrees) + " degrees", driving_data.ys[i])
    resFile.write(`i` + "," + `degrees` + "," + `driving_data.ys[i]` + "\n")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1 
    '''

    # we first need to obtain the steering angle in the fault-free run
    fi.turnOffInjections()
    #degrees = model.y.eval(feed_dict={model.x: [image]})[0][0] * 180.0 / scipy.pi 

    steering = sess.run(
                    model.steering,
                    feed_dict={
                        model.image_input: [image],
                        model.keep_prob: 1.0
                    }
                )

    degrees = steering[0][0] * 180.0 / scipy.pi

    golden = degrees 
    resFile.write("golden=" + str(golden) + ",")

    # perform FI
    fi.turnOnInjections() 
 
    totalFI = 0.
    sdcCount = 0  
    numOfInjection = 1
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
            print("steering is None")
            #degrees = model.y.eval(feed_dict={model.x: [image]})[0][0] * 180.0 / scipy.pi 
        else:
            degrees = steering[0][0] * 180.0 / scipy.pi

        totalFI += 1
        # we store the value of the deviation, so that we can use different thresholds to parse the results
        resFile.write(str(abs(degrees - golden)) + ",")
        print(i, totalFI)

    resFile.write("\n")

     
#cv2.destroyAllWindows()
