from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def predict(model, path, tags, train_transforms):
  with torch.no_grad():
    image = Image.open(path)
    image_t = train_transforms(image)
    image_t = torch.unsqueeze(image_t, 0)
    probs = torch.exp(model(image_t.to('cuda'))).cpu().numpy()
    print(tags[np.argmax(probs)])
    print(probs)
    return image


def calculate_confusion_matrix(model, testloader, device= torch.device('cuda')):
  true_labels = np.array([])
  pred_labels = np.array([])
  with torch.no_grad():
    for inputs, labels in testloader:
      inputs, labels = inputs.to(device), labels.to(device)
      logps = model.forward(inputs)
      ps = torch.exp(logps)
      top_p, top_class = ps.topk(1, dim=1)
      pred = top_class.cpu().numpy().ravel()
      pred_labels = np.hstack((pred_labels,pred))
      true_labels = np.hstack((true_labels, labels.cpu().numpy().ravel()))
  return confusion_matrix(true_labels, pred_labels)


def calculate_embeddings(model, loader, device=torch.device('cuda')):
  embeddings = []
  tags = []
  with torch.no_grad():
    for inputs, labels in loader:
      
      inputs, labels = inputs.to(device), labels.to(device)
      embedding = model.forward(inputs).cpu().numpy()
      embeddings.extend(embedding)
      tags.extend(labels.cpu().numpy())
  df = pd.DataFrame(embeddings)
  df['tag'] = tags
  return df


def scatter_plot(x, colors):
    # choose a color palette with seaborn.
    labels = list(np.unique(colors))
    colors_to_num = list(map(lambda x: labels.index(x), colors))
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors_to_num])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in np.unique(colors):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, i, fontsize=11)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts



