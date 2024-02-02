import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.optim as optim
from torchvision import transforms, models
from tqdm import trange
from torchvision.transforms.functional import rotate
# %matplotlib inline

class StyleTransfer:

    def __init__(self,content_image,style_image,style_image_2=None , model = None, alpha = 1, beta = 1e5,gamma=1e5,rotation=0 ) -> None:
        if model is not None:
            self.set_model(model)
        else:
            self.get_vgg()

        print("Model Fetched")

        if isinstance(content_image, str):
            self.content_image = self.load_images(content_image).to(self.device)
        else:
            self.content_image = content_image

        if isinstance(style_image, str):
            self.style_image = self.load_images(style_image,  shape=self.content_image.shape[-2:]).to(self.device)
        else:
            self.style_image = style_image
        
        if style_image_2:
            if isinstance(style_image, str):
                self.style_image_2 = self.load_images(style_image_2,  shape=self.content_image.shape[-2:]).to(self.device)
            else:
                self.style_image_2 = style_image
        else:
            self.style_image_2=None
        
        self.alpha = alpha
        self.beta = beta
        self.gamma=gamma
        self.rotation=rotation
        self.lr = 0.003
        self.content_loss_layer = 'conv4_2'
        self.loss = []

        self.layers = {'0': 'conv1_1',
                    '5': 'conv2_1',
                    '10': 'conv3_1',
                    '19': 'conv4_1',
                    '21': 'conv4_2',  ## content representation
                    '28': 'conv5_1'}
        
        self.layers_weights = {'conv1_1': 1.,
                 'conv2_1': 1,
                 'conv3_1': 1,
                 'conv4_1': 1,
                 'conv5_1': 1}
        self.exp_name = "Default"
        pass

    def set_model(self, model):
        self.model = model
        for param in self.model.parameters():
          param.requires_grad_(False)
        self.set_device()

    def set_layers(self, layers):
        self.layers = layers

    def set_layer_weights(self, weights):
        self.layers_weights = weights

    def set_content_loss_layer(self, layer):
        self.content_loss_layer = layer

    def set_lr(self, lr):
        self.lr = lr

    def get_vgg(self):
        self.model = models.vgg19(pretrained=True).features

        # freeze all VGG parameters since we're only optimizing the target image
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.set_device()
        return self.model

    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def set_exp(self, exp):
        self.exp_name = exp

    def load_images(self,img_path, max_size=400, shape=None):
        image = Image.open(img_path).convert('RGB')

        # large images will slow down processing
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)

        if shape is not None:
            size = shape

        in_transform = transforms.Compose([
                            transforms.Resize(size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))])

        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:3,:,:].unsqueeze(0)
        return image

    def get_loss(self):
        return self.loss

    def im_convert(self, tensor):
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1,2,0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)

        return image
    
    def get_images(self):
        return self.content_image.images, self.style_image.images
    
    def set_layers_weights(self, layers):
        self.layers_weights = layers

    def set_g_matrix(self, matrix):
        self.g_matrix = matrix

    def set_alpha_beta(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    
    def show_images(self):
        if self.style_image_2 is not None:
            fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(30, 10))
            ax1.imshow(self.im_convert(self.content_image))
            ax2.imshow(self.im_convert(self.style_image))
            ax3.imshow(self.im_convert(self.style_image_2))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
            ax1.imshow(self.im_convert(self.content_image))
            ax2.imshow(self.im_convert(self.style_image))
    
    def get_features(self, image, model = None, layers = None):
        """ Run an image forward through a model and get the features for
            a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
        """

        if model is None:
            model = self.model

        if layers is None:
            layers = self.layers

        features = {}
        x = image

        # model._modules is a dictionary holding each module in the model
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features
    
    def get_gram_matrix(self, tensor,is_target=False):
 
        _,depth,height, width = tensor.size()
 
        # reshape so we're multiplying the features for each channel
        
        if is_target and self.rotation != 0:
            tensor=rotate(tensor,self.rotation)
        
        tensor = tensor.view(depth, height * width)
        
       
        # calculate the gram matrix
        gram = torch.mm(tensor, tensor.t())
        
        return gram
    
    def run(self, show_every = 1000, steps = 5000, alpha = None, beta = None, gamma = None, opt = "Adam", output = "Outputs", show_results = False):

        output_path = output + os.sep + self.exp_name
        if not os.path.exists(output_path):
            # If it doesn't exist, create it along with any necessary parent directories
            os.makedirs(output_path)
        
        content_features = self.get_features(self.content_image, self.model)
        style_features = self.get_features(self.style_image, self.model)

        
        # calculate the gram matrices for each layer of our style representation
        style_grams = {layer: self.get_gram_matrix(style_features[layer]) for layer in style_features}

        if self.style_image_2 is not None:
            style_features_2=self.get_features(self.style_image_2, self.model)
            style_grams_2 = {layer: self.get_gram_matrix(style_features_2[layer]) for layer in style_features_2}
        # create a third "target" image and prep it for change
        # it is a good idea to start of with the target as a copy of our *content* image
        # then iteratively change its style
        target = self.content_image.clone().requires_grad_(True).to(self.device)

        style_weights = self.layers_weights

        # you may choose to leave these as 
        if alpha is None:
            content_weight = self.alpha  # 
        else:
            content_weight = alpha 

        if beta is None:
            style_weight = self.beta
        else:
            style_weight = beta
        
        if gamma is None:
            style_weight_2 = self.gamma
        else:
            style_weight_2 = gamma


        # iteration hyperparameters
        if opt == "Adam":
            optimizer = optim.Adam([target], lr=self.lr)
        if opt == "LBFGS":
            optimizer = optim.LBFGS([target], lr=self.lr)#, line_search_fn='strong_wolfe'

        print("Starting Training")
        for ii in trange(1, steps+1):

            ## TODO: get the features from your target image
            ## Then calculate the content loss

            

            target_features = self.get_features(target, self.model)

            content_loss =torch.mean((target_features[self.content_loss_layer] - content_features[self.content_loss_layer])**2)

            # the style loss
            # initialize the style loss to 0
            style_loss = 0
            style_loss_2=0
            # iterate through each style layer and add to the style loss
            for layer in style_weights:
                # get the "target" style representation for the layer
                target_feature = target_features[layer]
                target_gram = self.get_gram_matrix(target_feature,is_target=True)
                _, d, h, w = target_feature.shape

                #get the "style" style representation
                style_gram = style_grams[layer]
        
            ##  Calculate the style loss for one layer, weighted appropriately
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
                # add to the style loss
                style_loss += layer_style_loss / (d * h * w)

                
                if self.style_image_2 is not None:
                    style_gram_2= style_grams_2[layer]
                    layer_style_loss_2 = style_weights[layer] * torch.mean((target_gram - style_gram_2)**2)
                    style_loss_2 += layer_style_loss_2 / (d * h * w)

                
            ## TODO:  calculate the *total* loss
            self.total_loss = content_weight * content_loss + style_weight * style_loss + style_weight_2*style_loss_2
            total_loss = self.total_loss
            self.loss.append(total_loss)
            # update your target image
            optimizer.zero_grad()
            total_loss.backward()
            self.total_loss = total_loss
            def closure():
                return self.total_loss

            if opt == "LBFGS":
                optimizer.step(closure)
            else:
                
                optimizer.step()

            # display intermediate images and print the loss
            if ii % (steps/100) == 0:
                plt.imsave( output_path + "/target_image_{:06d}.png".format(ii), self.im_convert(target))
            if show_results or ii == steps:
                if  ii % show_every == 0 or ii == steps:
                    print("Epoch: ",ii)
                    print('Total loss: ', total_loss.item())
                    plt.imshow(self.im_convert(target))
                    plt.show()

def create_gif(exp_name = "Default"):
  # Retrieve image file names
  folder_path = "Outputs/"+exp_name
  images = [img for img in os.listdir(folder_path) if img.endswith(".png")]
  images.sort()
  frame_one = Image.open(os.path.join(folder_path, images[0]))
  frame_others = [Image.open(os.path.join(folder_path, img)) for img in images[1:]]
  frame_one.save("Outputs/{}/gif.gif".format(exp_name), format='GIF', append_images=frame_others,
                save_all=True, duration=100, loop=0)

        
