import webbrowser
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision

from PIL import Image, ImageFile
from torch import nn
from torch import optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models, transforms

import cv2

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')

else:
    print('CUDA is available!  Training on GPU ...')
ImageFile.LOAD_TRUNCATED_IMAGES = True
#!pip install --upgrade wandb




test_transforms = transforms.Compose([transforms.Resize(255),
                                      #  transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      ])

model = models.densenet161()


model.classifier = nn.Sequential(nn.Linear(2208, 1000),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1000, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = model.cuda()

model.load_state_dict(torch.load('tensorboardexp.pt'))
classes = ["accident", "noaccident"]
# model.load_state_dict(torch.load('tensorboardexp.pt'))
count = 0
counts = 1
videopath = 'Both.mp4'



vid = cv2.VideoCapture(videopath)
ret = True
while ret:
    if ret == True:
        ret, frame = vid.read()

        try:
            img = Image.fromarray(frame)
        except ValueError:
            break
        except AttributeError:
            break
        img = test_transforms(img)
        img = img.unsqueeze(dim=0)
        img = img.cuda()
        model.eval()
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)

            index = int(predicted.item())
            if index == 0: 
                cv2.imwrite(r"D:\xampp\htdocs\accident\frame%d.png" % count, frame)
                count += 1
                # Download the helper library from https://www.twilio.com/docs/python/install
                import os
                from twilio.rest import Client

                account_sid = "AC4715e7b764418486e2e74bb02987cadd"
                auth_token = "8febad04fc044b6d435dbf662844baf8"
                client = Client(account_sid, auth_token)
                message = client.messages.create(body='"Alert!!"  "Alert!!" Accident Detected.',from_="+12545363940",to="+918374131403")
                print(message.sid)

                

                #check
                def get_location():
                    geolocator = Nominatim(user_agent="your_app_name")

                    try:
                        location = geolocator.geocode(None, timeout=10)
                        return location
                    except GeocoderTimedOut:
                        return get_location()  # Retry in case of timeout

                def send_location_sms(location):
                    client = Client(account_sid, auth_token)
                    google_maps_link = f'https://www.google.com/maps?q={location.latitude},{location.longitude}'

                    body = f'Google Maps Location: {google_maps_link}'

                    message = client.messages.create(body=body,from_="+12545363940",to="+918374131403")

                    print(f'SMS sent. Message SID: {message.sid}')
                    #check
                location = get_location()
                if location is not None:
                    send_location_sms(location)
                else:
                    print('Failed to obtain location')
                        ##check
                    
                if counts == 1:

                    ##chack
                    # Add this code inside the loop where the accident is detected
                    

                    chrome_path = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s"
                    browser = webbrowser.get(chrome_path)
                    browser.open('localhost',new=2)

                    #webbrowser.open('localhost', new=2)
                    counts += 1
                    
           

            labels = 'status: ' + classes[index]

        cv2.putText(frame, labels, (10, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv2.destroyAllWindows()
