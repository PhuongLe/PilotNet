import tensorflow as tf
import carla

import scipy.misc
from nets.pilotNet import PilotNet
import cv2
from subprocess import call
import random
import time
import numpy as np
from multiprocessing import Queue

FLAGS = tf.app.flags.FLAGS

"""model from nvidia's training"""

# generated model after training
tf.app.flags.DEFINE_string(
    'model', './data/models/model.ckpt',
    """Path to the model parameter file.""")

tf.app.flags.DEFINE_string(
    'steer_image', './data/.logo/steering_wheel_image.jpg',
    """Steering wheel image to show corresponding steering wheel angle.""")

WIN_MARGIN_LEFT = 240
WIN_MARGIN_TOP = 240
WIN_MARGIN_BETWEEN = 180

IM_WIDTH = 455
IM_HEIGHT = 256

counter = 0
#counterMod = 0
imageQueue = Queue(maxsize = 100) 

def process_img(image):
    global counter
    global imageQueue

    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    imageQueue.put(i3)
    #cv2.imwrite('./output1/{0}.jpg'.format(counter), i3)
    counter += 1   

actor_list = []

if __name__ == '__main__':
    print("hello")
    img = cv2.imread(FLAGS.steer_image, 0)
    rows,cols = img.shape

    # Visualization init
    cv2.namedWindow("Steering Wheel", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Steering Wheel", WIN_MARGIN_LEFT, WIN_MARGIN_TOP)
    cv2.namedWindow("camera", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow("camera", IM_WIDTH, IM_HEIGHT)
    cv2.moveWindow("camera", WIN_MARGIN_LEFT+cols+WIN_MARGIN_BETWEEN, WIN_MARGIN_TOP)
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        world = client.get_world()

        blueprint_library = world.get_blueprint_library()

        bp = blueprint_library.filter('model3')[0]
        print(bp)

        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

        actor_list.append(vehicle)
        
        # get the blueprint for this sensor
        blueprint = blueprint_library.find('sensor.camera.rgb')
        # change the dimensions of the image
        blueprint.set_attribute('image_size_x', '{w}'.format(w=IM_WIDTH))
        blueprint.set_attribute('image_size_y', '{h}'.format(h=IM_HEIGHT))
        blueprint.set_attribute('fov', '110')
        #blueprint.set_attribute('sensor_tick', '0.1')

        # Adjust sensor relative to vehicle
        transform = carla.Transform(carla.Location(x=2.5, z=0.75))

        # spawn the sensor and attach to vehicle.
        sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
        
        # add sensor to list of actors
        actor_list.append(sensor)

        # do something with this sensor
        #sensor.listen(lambda image: image.save_to_disk('output/%.6d.jpg' % image.frame))
        sensor.listen(lambda data: process_img(data))

        #time.sleep(0.1)  

        with tf.Graph().as_default():
            smoothed_angle = 0
            i=0
            # construct model
            model = PilotNet()

            saver = tf.train.Saver()
            with tf.Session() as sess:
                # restore model variables
                saver.restore(sess, FLAGS.model)

                while(cv2.waitKey(10) != ord('q')):
                    while i>=counter:
                        time.sleep(0.01)  
                    
                    #full_image = scipy.misc.imread("./output1" + "/" + str(i) + ".jpg", mode="RGB")                    
                    full_image = imageQueue.get()
                    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
                    steering = sess.run(
                        model.steering,
                        feed_dict={
                            model.image_input: [image],
                            model.keep_prob: 1.0
                        }
                    )

                    degrees = steering[0][0] * 180.0 / scipy.pi
                    #degrees = 0.1
                    call("clear")
                    #print("Queue size = {0}. Rending image..{1}".format(imageQueue.qsize(), i))
                    print("Predicted steering angle: " + str(degrees) + " degrees")

                    # convert RGB due to dataset format
                    cv2.imshow("camera", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
                    print("camera image size: {} x {}").format(full_image.shape[0], full_image.shape[1])

                    # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
                    # and the predicted988j98u4 angle
                    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
                    M = cv2.getRotationMatrix2D((cols/2,rows/2), -smoothed_angle, 1)
                    dst = cv2.warpAffine(img,M,(cols,rows))
                    cv2.imshow("Steering Wheel", dst)
                    #if i>100:
                    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=degrees))

                    i += 1
    finally:
        cv2.destroyAllWindows()
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')
