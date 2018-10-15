from collections import OrderedDict
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torchvision import models, transforms
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from skimage import io



class _PropagationBase(object):
    def __init__(self, model):
        super(_PropagationBase, self).__init__()
        self.model = model
        self.image = None

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot

    def forward(self, image):
        self.image = image.requires_grad_()
        self.model.zero_grad()
        self.preds = self.model(self.image)
        self.probs = F.softmax(self.preds, dim=1)[0]
        self.prob, self.idx = self.probs.sort(0, True)
        return self.prob, self.idx

    def backward(self, idx):
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)

class GradCAM(_PropagationBase):
    def __init__(self, model):
        super(GradCAM, self).__init__(model)
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.detach()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        #ReLU(sum(grad_weight on channel k*feature map channel k))
        gcam = (fmaps[0] * weights[0]).sum(dim=0)
        #ReLU
        gcam = torch.clamp(gcam, min=0.)

        gcam -= gcam.min()
        gcam /= gcam.max()

        return gcam.detach().cpu().numpy()
    
#given a model name (must be in the CONFIG dict), and image, and the top number of predictions you want to get, returns the gradcam object 
def get_gradCAM_img(image_path, model_name, categories, alternate_layer = None):
    CONFIG = {
        'resnet152': {
            'target_layer': 'layer4.2',
            'input_size': 224
        },
        'vgg19': {
            'target_layer': 'features.36',
            'input_size': 224
        },
        'vgg19_bn': {
            'target_layer': 'features.52',
            'input_size': 224
        },
        'inception_v3': {
            'target_layer': 'Mixed_7c',
            'input_size': 299
        },
        'densenet201': {
            'target_layer': 'features.denseblock4',
            'input_size': 224
        },
        'resnet18': {
            'target_layer': 'layer4',
            'input_size': 224
        },
        # Add your model
    }.get(model_name)

    #device = torch.device('cpu')

    # Synset words
    classes = list()
    with open('samples/synset_words.txt') as lines:
        for line in lines:
            line = line.strip().split(' ', 1)[1]
            line = line.split(', ', 1)[0].replace(' ', '_')
            classes.append(line)

    # Model
    model = models.__dict__[model_name](pretrained=True)
    #model.to(device)
    model.eval()

    # Image
    raw_image = cv2.imread(image_path)[..., ::-1]
    raw_image = cv2.resize(raw_image, (CONFIG['input_size'], ) * 2)
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])(raw_image).unsqueeze(0)
    
    gcam = GradCAM(model=model)
   # probs, idx = gcam.forward(image.to(device))
    probs, idx = gcam.forward(image)

    for i in range(0, categories):
        gcam.backward(idx=idx[i])
        #if an alternate layer is input, use that instead of last layer
        if alternate_layer:
            output = gcam.generate(target_layer=alternate_layer)
        else:
            output = gcam.generate(target_layer=CONFIG['target_layer'])
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        result = cv2.resize(output, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(result * 255.0), cv2.COLORMAP_JET)
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        plt.imshow(cam)