from torchvision import datasets, models, transforms
# import mb2
import cv2
import numpy as np
import torch
import requests
import os
import shutil

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=torch.load('model/savetest.pth',map_location=DEVICE)


dir = "./test"

model.eval()

if torch.cuda.is_available():
    model.to('cuda')
    device = 'cuda'
else:
    device = 'cpu'

classes = ['balloon', 'banana', 'bell', 'cdplayer', 'cleaver',
 'cradle', 'crane', 'daisy', 'helmet', 'speaker']

for image in os.listdir(dir):
    if os.path.isfile(os.path.join(dir,image)):
        orig_image1 = cv2.imread(dir+"/"+image)
        image_name = image

        to_pil = transforms.ToPILImage()
        orig_image = to_pil(orig_image1)
        trans = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor()
                                ,transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])

        image = trans(orig_image)
        image = image.unsqueeze(0)
        image = image.to(device)

            
        with torch.no_grad():
            result = model(image)

        print("-"*50)
        pr = torch.argmax(torch.nn.functional.softmax(result[0], dim=0))
        result1 = torch.nn.functional.softmax(result[0], dim=0)
        round_result = round(float(result1[pr]),4)
        print(f"conf : {round_result}, result : {pr}")


        shutil.copy(dir+"/"+image_name, dir+"/"+ classes[pr] +"/"+image_name)





