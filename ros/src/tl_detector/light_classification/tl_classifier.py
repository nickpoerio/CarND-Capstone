from styx_msgs.msg import TrafficLight
from keras.models import load_model
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model_sim = load_model('light_classification/TL_sim_classifier.h5')
        self.model_site = load_model('light_classification/TL_site_classifier.h5')
    

    def get_classification(self, image, is_site):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        np_image = np.expand_dims(image, axis=0)
        
        if is_site is True:
            out=self.model_site.predict(np_image)
        else:
            out=self.model_sim.predict(np_image)
            
        if out>.5:
            return TrafficLight.RED
        
        return TrafficLight.UNKNOWN
        
