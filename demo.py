from model import *

import cvfy
import cv2
app = cvfy.register("APP TOKEN")


@cvfy.crossdomain
@app.listen()
def image_captioning():

    image_paths = cvfy.getImageArray()
    print image_paths
    caption = test_tf(image_paths[0])
    #print caption
    cvfy.sendTextArray([caption])
    return 'OK'

app.run()
