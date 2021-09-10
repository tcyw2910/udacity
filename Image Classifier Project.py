#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports here
import torch
from torchvision import datasets, transforms

from torch import nn, optim
import torch.nn.functional as F

import matplotlib.pyplot as plt


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])

validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)

validation_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)

test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = 64)

test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[4]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# def check_gpu(gpu_arg):
#     if not gpu_arg:
#         return torch.device("cpu")
#     
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     
#     if device == "cpu":
#         print("CUDA wasn't found. Will use CPU.")
#     return device

# In[5]:


# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load a pre-trained network
import torchvision.models as models

model = models.vgg16(pretrained = True)
model


# In[6]:


# FREEZE parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 500)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.05)),  
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


model.classifier = classifier


# In[7]:


# Define the criterion and optimizer

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are FROZEN
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)

model.to(device);


# In[8]:


# Train the Network here
epochs = 2
steps = 0
running_loss = 0
print_every = 50

for epoch in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        steps += 1
        
        # move images and label tensors to the default device 
        images, labels = images.to(device), labels.to(device)
        
        # training pass
        optimizer.zero_grad()
        
        # make a forward pass
        output = model.forward(images)
        
        # calculate loss using the logits
        loss = criterion(output, labels)
        
        # perform a backward pass
        loss.backward()
        
        # take a step w/ optimizer to update the weights
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate Accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.."
                  f"Train Loss: {running_loss/print_every:.3f}.."
                  f"Test Loss: {test_loss/len(validation_loader):.3f}.."
                  f"Test Accuracy: {accuracy/len(validation_loader):.3f}")
            
            running_loss = 0
            model.train()


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[9]:


# TODO: Do validation on the test set
accuracy = 0
test_loss = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logps = model.forward(images)
        
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()
        
        # Calculate Accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Test Loss: {test_loss/len(test_loader):.3f}.."
              f"Test Accuracy: {accuracy/len(test_loader):.3f}")


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[10]:


# TODO: Save the checkpoint 
# torch.save(model.state_dict(), 'checkpoint.pth')

# Assign Class_to_idx as attribute to model
model.class_to_idx = train_data.class_to_idx

# Define checkpoint w/ parameters to be saved
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'classifier': model.classifier,
              'epochs': 2,
              'hidden_layers': 4096,
              'learning_rate': 0.003,
              'arch': "vgg16",
              'optimizer_dict': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict()
             }

# Save checkpoint
torch.save(checkpoint, 'checkpoint.pth')




# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[14]:


# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    
    model.load_state_dict(checkpoint['state_dict'])
    
    learning_rate = checkpoint['learning_rate']
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    
    return optimizer, model

optimizer, model = load_checkpoint('checkpoint.pth')


# In[15]:


print(model)


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[16]:


from PIL import Image
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    #method1
    
    # from PIL import image path
    im = Image.open(image)
    
    # DO NOT USE FOR NOW!!!
    #width, height = im.size
    
    #if width > height:
    #    ratio = width/height
    #    im.thumbnail((ratio*256, 256))
    #elif height > width:
    #   im.thumbnail((256, height/width*256))
    
    w = 256
    h = 256
    im = im.resize((w, h))
    
    new_width = 224
    new_height = 224
    
    left = (w - new_width)/2
    top = (h - new_height)/2
    right = (w + new_width)/2
    bottom = (h + new_height)/2
    
        
    #new_width, new_height = im.size 
    
    #left = (new_width - 224)/2
    #top = (new_height - 224)/2
    #right = (new_width + 224)/2
    #bottom = (new_height + 224)/2
    im = im.crop((left, top, right, bottom))
    
    # Normalize
    np_image = np.array(im)/255
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    
    # subtract means and divide by standard deviation
    np_image = (np_image - mean)/std
    
    # transpose image
    image = np_image.transpose((2, 0, 1))
    
    return image

#image = test_dir + '/1/image_06743.jpg'
#process_image(image)
    


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[19]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[20]:


imshow(process_image("flowers/test/1/image_06743.jpg"));


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[21]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    ##image = process_image(image_path).type(torch.FloatTensor).unsqueeze(0).to(device)
    image = torch.tensor(process_image(image_path)).type(torch.FloatTensor).unsqueeze(0).to(device)
    
    load_checkpoint('checkpoint.pth')
    
    model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    
    ps_topk_class = []
    
    with torch.no_grad():
        #image = process_image(image_path).unsqueeze(dim=0)
        outputs = model.forward(image)
        ps = torch.exp(outputs)
        
        # takes top 5 probabilities and index from the output
        ps_topk = ps.cpu().topk(topk)[0].numpy()[0]
        ps_topk_index = ps.cpu().topk(topk)[1].numpy()[0]
        
        # Loop through class_to_idx dictionary to reverse key, values in idx_to_class dictionary
        for key, value in model.class_to_idx.items():
            model.idx_to_class[value] = key
        
        # Loop through index to retrieve class from idx_to_class dict
        for item in ps_topk_index:
            ps_topk_class.append(model.idx_to_class[item])
            
    return ps_topk, ps_topk_class
        
        #probs, indices = ps.topk(topk)
        #probs = probs.squeeze()
        #classes = [model.idx_to_class[idx] for idx in indices[0].tolist()]
        
    #return probs, classes #previously was probs, top_classes
    
   # print(predict('sample_img.jpg', model))


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# # import matplotlib.pyplot as plt
# 
# 
# # TODO: Display an image along with the top 5 classes
# def sanity_check():
#     plt.rcParams["figure.figsize"] = (10, 5)
#     plt.subplot(211)
#     
#     index = 1
#     path = "flowers/test/1/image_06743.jpg"
#     
#     image = process_image(path)
#     
#     probabilties = predict(path, model)
#     
#     axs = imshow(image, ax = plt)
#     axs.axis('off')
#     axs.title(cat_to_name[str(index)])
#     axs.show()
#     
#     
#     a = np.array(probabilities[0][0])
#     b = [cat_to_name[str(index+1)] for index in np.array(probabilities[1][0])]
#     
#     
#     N=float(len(b))
#     fig,ax = plt.subplots(figsize=(8,3))
#     width = 0.8
#     tickLocations = np.arange(N)
#     ax.bar(tickLocations, a, width, linewidth=4.0, align = 'center')
#     ax.set_xticks(ticks = tickLocations)
#     ax.set_xticklabels(b)
#     ax.set_xlim(min(tickLocations)-0.6,max(tickLocations)+0.6)
#     ax.set_yticks([0.2,0.4,0.6,0.8,1,1.2])
#     ax.set_ylim((0,1))
#     ax.yaxis.grid(True)
#     
#     plt.show()

# In[22]:


import seaborn as sns

image = test_dir + '/1/image_06743.jpg'

def sanity_check(image, model):
    
    tensor_image = process_image(image)
    ps_topk, ps_topk_class = predict(image, model, topk = 5)
    ps_topk, ps_topk_class
    
    probs = np.flip(ps_topk, axis = 0)
    classes = np.flip(ps_topk_class, axis = 0)
    labels = [c1 for c1 in classes]
    
    np_image = process_image(image)
    np_image = np_image.squeeze()
    img_tensor = torch.from_numpy(np_image)
    
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    
    flower_num = image.split('/')[2]
    title = cat_to_name[flower_num]
    
    ax.set_title(title)
    imshow(img_tensor, ax);
    class_idx = model.class_to_idx
    
    plt.subplot(2,1,2)
    sns.barplot(x=probs*100, y=labels, color=sns.color_palette()[0]);
    
    plt.show()
    plt.tight_layout


# In[23]:


sanity_check(image, model)


# In[ ]:





# In[ ]:





# In[ ]:




