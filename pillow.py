from PIL import Image,ImageDraw,ImageFont
import time

import pandas as pd
import numpy as np
# ######################################################
def get_rotated(data): 
    
    image_data = data[:, :-1]
    rotations = data[:, -1].astype(int)
    
    ## rotate each row
    for i in range(image_data.shape[0]):
        shift = -rotations[i]
        image_data[i] = np.roll(image_data[i], shift)
    
    return image_data

all_data = np.array(np.genfromtxt("traindata.txt", delimiter=","))
all_data = get_rotated(all_data)

#############################################
data = pd.DataFrame(all_data)


for i in range(len(all_data)):
    data = pd.DataFrame(all_data)
    data = data.to_numpy()
    data = data[i].reshape(26,40)

    data = data.astype(np.uint8)
    img = Image.fromarray(data)
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default()
    text = i

    draw.text((10,10), text,fill=(0,0,0), font=font)
# Set pixel values based on the data

# Show the image
    img.show()
    time.sleep(0.05)



