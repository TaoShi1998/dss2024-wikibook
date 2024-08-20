# Learning Transferable Visual Models From Natural Language Supervision

## Overview

This chapter explores the method of learning transferable visual models from natural language supervision, a groundbreaking approach that leverages large-scale natural language data to train visual models. This method has been shown to achieve state-of-the-art performance on various vision tasks by effectively transferring knowledge from textual descriptions to visual understanding.

### Learning goals:

In this chapter, we will cover:

1. The fundamental concepts of learning from natural language supervision.
2. The architecture and training process of models using this approach.
3. Mathematical formulation and objective functions used in this method.
4. Practical implementation tips and common pitfalls.
5. Applications and advancements derived from this approach.

## Background

To understand this method, a solid grasp of neural networks, deep learning fundamentals, and natural language processing (NLP) is required. Prior works that laid the groundwork for this method include advancements in transfer learning, word embeddings, and image classification. Key references include:

- A Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision" [[1]](#1)


For a more in-depth review, readers can refer to external resources such as the BERT and CLIP papers.

## Problem Formulation & Method Explanation

### Problem Formulation

The goal is to train visual models that can effectively transfer knowledge from natural language descriptions to visual understanding. This involves two main components:

- **Text Encoder**: Converts natural language descriptions into dense vector representations.
- **Image Encoder**: Converts images into dense vector representations.

The objective is to align these representations in a shared latent space such that corresponding images and text descriptions are close together.

### Method Explanation

1. **Text Encoder**: Typically a transformer-based model like BERT or a similar architecture that encodes textual data into a dense vector space.
2. **Image Encoder**: Typically a convolutional neural network (CNN) or vision transformer (ViT) that encodes visual data into a dense vector space.

### Training Process

1. **Initialize** the weights of both the text encoder and the image encoder.
2. **For each training iteration**:
   - **Compute Text Embeddings**: Pass text descriptions through the text encoder.
   - **Compute Image Embeddings**: Pass corresponding images through the image encoder.
   - **Compute Loss**: Use a contrastive loss function to maximize the similarity of corresponding text and image pairs while minimizing the similarity of non-corresponding pairs.
3. **Update Weights**: Use backpropagation to update the weights of both encoders based on the computed loss.

The training process involves alternating between computing embeddings and updating model weights to improve alignment in the shared latent space.

## Code Example

Below is a simplified implementation of this method using PyTorch. The full demo code, including setup instructions, can be found in the accompanying zip archive.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from torchvision import models, transforms

# Define the text encoder
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

# Define the image encoder
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 768)
    
    def forward(self, images):
        return self.resnet(images)

# Define the contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, image_embeddings, text_embeddings):
        scores = torch.matmul(image_embeddings, text_embeddings.T)
        labels = torch.eye(image_embeddings.size(0)).to(image_embeddings.device)
        loss = nn.CrossEntropyLoss()(scores, labels.argmax(dim=1))
        return loss

# Hyperparameters
batch_size = 32
learning_rate = 0.0001
epochs = 10

# Initialize models and tokenizer
text_encoder = TextEncoder()
image_encoder = ImageEncoder()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Optimizers
optimizer = optim.Adam(list(text_encoder.parameters()) + list(image_encoder.parameters()), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        images, captions = batch
        inputs = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        images = images.to(device)

        # Compute embeddings
        text_embeddings = text_encoder(input_ids, attention_mask)
        image_embeddings = image_encoder(images)

        # Compute loss
        loss = ContrastiveLoss()(image_embeddings, text_embeddings)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}] Loss: {loss.item()}')

print("Training finished.")
```


We can slightly modify the encoder part, and build autoencoder:

```python
class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)
```

## Discussion

Discussion on the chosen representation learning method, including but not limited to:

* When to use this method (advantage and limitation)
* Practical tips in implementation and usage
* Relationship to other methods, especially those covered in class.
* Important subsequent representation learning algorithms derived from this approach
* Impactful application works using this method

### When to use this method

- **Advantages**: This method leverages large-scale natural language data, which is abundant and diverse, allowing models to learn rich and transferable visual representations.
- **Limitations**: Requires significant computational resources for training and may face challenges with aligning very different types of data (e.g., text and images).

### Practical Tips

- Ensure the text and image encoders are properly pre-trained on large datasets.
- Use a large and diverse dataset to cover a wide range of visual and textual concepts.
- Regularly evaluate the model on downstream tasks to monitor transferability and generalization.

### Relationship to Other Methods

- Compared to traditional supervised learning, this method uses natural language as a supervisory signal, which can be more flexible and scalable.
- Similar to methods like CLIP (Contrastive Language-Image Pre-Training), which also aligns text and image representations in a shared latent space.

### Subsequent Algorithms

- **CLIP (Contrastive Language-Image Pre-Training)**: A significant advancement that leverages a similar approach to achieve state-of-the-art performance on various vision tasks.
- **ALIGN (A Large-scale ImaGe and Noisy-text embedding)**: Another method that extends this approach to handle noisy text data and large-scale image datasets.

### Applications

- **Zero-shot Learning**: Applying the model to new tasks without additional training.
- **Image Search**: Retrieving images based on textual descriptions.
- **Visual Question Answering**: Using the learned representations to answer questions about images.

## References
<a id="1">[1]</a> A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, "Learning Transferable Visual Models From Natural Language Supervision," in *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139, 2021, pp. 8748-8763.

## Author Team

**Wu Changhao**
