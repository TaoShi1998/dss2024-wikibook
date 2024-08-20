# Diffusion Models



## 1. Overview

Diffusion models have emerged as a highly influential and impactful technique for image generation over the past decade. They outperform the previous state-of-the-art GANs across many benchmarks and become the preferred choice for generative modeling, especially in visual domains. Notable examples include Google's ImageGen for text-to-image generation and OpenAI's DALL.E 2. Recently, the application of diffusion models has expanded dramatically across various tasks, mirroring the widespread adoption of GANs seen between 2017 and 2020 [[1]](#1).

Many of the fundamental concepts behind diffusion models are similar to those found in earlier generative models we've discussed in this book, such as generative adversarial network (GANs) and variational autoencoders (VAEs). Since the existing generative models often suffer from training instability and provide approximate likelihoods, developing diffusion models to generate high-quality samples and offer more stable training dynamics is quite significant. 

In this **Chapter**, we will dive into the diffusion models, including the theory behind how they work and practical example of how to build the model.

In **Section 1**, we give a brief overview of the diffusion models and illustrate the learning goals of this chapter. This technique is based on the idea of iteratively adding noise to an image and then training a model to remove the noise, giving us the ability to transform pure noise into realistic samples.

**Section 2** introduces the background of the diffusion models, offering the mathematical concepts needed to understand the methodology of this model.

In **Section 3**, we provide a detailed explanation of diffusion models, covering problem formulation and derivation of solutions.

In **Section 4**, we walk through a sample implementation and usage demo of diffusion models.

**Section 5** focuses on discussion on diffusion models, covering when to use them, their relationship to other methods, derived algorithms, and impactful applications.

### Learning goals:

The topic you are presenting may be very big. Give a preview on what would be covered in this chapter (e.g. the scope). 

* Understand the underlying principles and components that define a diffusion model. 
* See how the forward process and reverse process are used in the model.
* Implement a demo of diffusion models.
* See the impactful application works related to diffusion models.

## 2. Background
 
Generative AI refers to a branch of artificial intelligence [[2]](#2) that focuses on creating models and algorithms capable of generating new, original content, such as images, text, music, and even videos. Unlike traditional AI models that are trained to perform specific tasks, generative AI models aim to learn and mimic patterns from existing data to generate new, unique outputs. For example, if I train a model on a dataset of images of cats, I can then use that model to generate new images of cats that look like they could have come from the original dataset. This is a powerful idea, and it has a wide range of applications, from creating novel images and videos to generating text with a specific style [[3]](#3). The two prominent generative models, namely, generative adversarial networks (GANs) and variational autoencoders (VAEs), have gained substantial recognition. GANs have exhibited versatility across various applications, yet their training complexity and limited output diversity, caused by challenges like mode collapse and gradient vanishing, have been evident. On the other hand, VAEs, while having a strong theoretical foundation, encounter difficulties in devising effective loss functions, resulting in suboptimal outputs.

Another category of techniques, inspired by probabilistic likelihood estimation and drawing parallels from physical phenomena, has emerged—these are known as diffusion models. Diffusion models are a relatively novel class of generative models that draw inspiration from physical processes like the diffusion of particles and concepts from information theory. They aim to generate data by iteratively transforming noise into structured information, essentially reversing the process of noise introduction. Indeed, the name 'diffusion' takes inspiration from the well-studied property of thermodynamic diffusion: an important link was made between this purely physical field and deep learning in 2015. [[4]](#4) Important progress was also being made in the field of score-based generative models,[[5]](#5) a branch of energy-based modeling that directly estimates the gradient of the log distribution (also known as the score function) in order to train the model, as an alternative to using contrastive divergence. In particular, Yang Song and Stefano Ermon [[6]](#6) used multiple scales of noise perturbations applied to the raw data to ensure the model—a noise conditional score network (NCSN)—performs well on regions of low data density. The breakthrough diffusion model paper came in the summer of 2020. [[7]](#7) Standing on the shoulders of earlier works, the paper uncovers a deep connection between diffusion models and score-based generative models, and the authors use this fact to train a diffusion model that can rival GANs across several datasets, called the Denoising Diffusion Probabilistic Model (DDPM). 

To fully understand diffusion models, it is essential to grasp several foundational mathematical and machine learning concepts.

### 2.1 Probability Theory

Probability theory is the branch of mathematics that deals with the analysis of random phenomena. It provides the foundational framework for understanding how likely events are to occur and is essential for modeling uncertainty. It underpins the noise addition and removal processes in diffusion models, where noise is typically added according to a specific probability distribution (e.g., Gaussian noise).

### 2.2 Stochastic Processes
A stochastic process is a collection of random variables indexed by time or space, representing systems that evolve over time in a probabilistic manner. Diffusion models often use stochastic processes to describe the evolution of data as noise is added (forward process) and removed (reverse process).

### 2.3 Variational Inference
Variational inference is a method for approximating complex probability distributions through optimization. It is commonly used in Bayesian statistics to approximate posterior distributions. Variational inference techniques can be used to optimize the parameters of diffusion models, particularly in Variational Diffusion Models (VDM).

### 2.4 Gradient Descent

Gradient descent is a fundamental optimization algorithm that iteratively adjusts model parameters in the direction of the steepest decrease in the loss function. Effective optimization is crucial for training diffusion models, ensuring that they learn to generate high-quality data by minimizing the loss associated with the noise addition and removal processes.

Understanding these foundational concepts in probability theory, stochastic processes, variational inference, optimization techniques, neural networks, and generative models is essential for fully understanding diffusion models.

## 3. Problem Formulation 

### 3.1 Forward Diffusion Process

The forward diffusion process transforms data $\mathbf{x_0}$ into a series of increasingly noisy versions $\mathbf{x_1}$, $\mathbf{x_2}$, $...$, $\mathbf{x_T}$. This process is defined by a Markov chain, where each step adds a small amount of Gaussian noise:

$$ 
q(x_t \mid \mathbf{x_{t-1}}) = \cal{N}(\mathbf{x_t}; \sqrt{1 - \beta_t} \mathbf{x_{t-1}}, \beta_t \textbf{I}) $$

Here, $\beta_t$ is a variance schedule, often linearly increasing from a small value to a larger value over time steps $t$. $\mathcal{N}$ denotes a normal (Gaussian) distribution.

The marginal distribution after $t$ steps can be derived as:

$$
q(x_t \mid x_0) = \cal{N}(\mathbf{x_t}; \sqrt{\bar{\alpha}_t} \mathbf{x_0}, (1 - \bar{\alpha}_t) \textbf{I}), $$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_ t = {\prod\limits_ {s = 1} ^ t} {\alpha_ s}$. This equation shows that $\mathbf{x}_t$ is a noisy version of $\mathbf{x}_0$, with noise increasing over time.


### 3.2 Reverse Diffusion Process
The reverse process aims to denoise $\mathbf{x}_ t$ to recover $\mathbf{x}_ {t-1}$, removing the noise added in the forward process. The reverse process is also represented as a Markov chain, but it works backwards:

$$
p_\theta(\mathbf{x}_ {t-1} | \mathbf{x}_ t) = \cal{N}(\mathbf{x}_ {t-1}; \mu_\theta(\mathbf{x}_ t, t), \Sigma_\theta(\mathbf{x}_ t, t)) 
$$

By minimizing the difference between the true reverse process and the learned reverse process, $\mu_\theta$ and $\Sigma_\theta$ are parameters of the Gaussian distribution, typically modeled by neural networks and learned in the training.

### 3.3 Training Procedure

* Start with a dataset of real samples $\mathbf{x}_0$.

* Add noise to the data to create noisy versions $\mathbf{x}_t$ at different steps.

* Use a neural network to predict the noise added at each step.

* Calculate the loss between the true noise and the predicted noise. Use a loss function to measure how well the model can predict the noise added at each step. A common loss function is:

$$
L(\theta) = \sum\limits_ {t=1}^T \mathbb{E}_ {q(\mathbf{x}_ t | \mathbf{x}_ 0)} \left[ \| \epsilon - \epsilon_\theta(\mathbf{x}_ t, t) \|^2 \right] 
$$

where $\epsilon$ is the true noise, $\epsilon_\theta$ is the predicted noise by the model.

* Adjust the model parameters to minimize the loss using gradient descent.

Diffusion models offer a robust framework for generative modeling by learning to reverse a noisy process. By carefully designing the forward and reverse processes and using effective training strategies, these models can generate high-quality samples from complex data distributions. Understanding the basic principles and mathematical foundations of diffusion models is key to leveraging their full potential in various applications.

## 4. Code Example

### 4.1 Environment Setup

First, we'll need to set up our Python environment. Here's a list of required packages:
- `pytorch`
- `torch`
- `torchvision`
- `math`
- `imageio`
- `glob`

We can install these packages using pip:

```bash
pip install pytorch torch torchvision math imageio glob
```


### 4.2 Define the Processes of the Model


```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from modules import *

class DiffusionModel(pl.LightningModule):
    def __init__(self, in_size, t_range, img_depth):
        super().__init__()
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.in_size = in_size

        bilinear = True
        self.inc = DoubleConv(img_depth, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, img_depth)
        self.sa1 = SAWrapper(256, 8)
        self.sa2 = SAWrapper(256, 4)
        self.sa3 = SAWrapper(128, 8)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)

    def forward(self, x, t):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 128, 16)
        x3 = self.down2(x2) + self.pos_encoding(t, 256, 8)
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + self.pos_encoding(t, 256, 4)
        x4 = self.sa2(x4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 128, 8)
        x = self.sa3(x)
        x = self.up2(x, x2) + self.pos_encoding(t, 64, 16)
        x = self.up3(x, x1) + self.pos_encoding(t, 64, 32)
        output = self.outc(x)
        return output

    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (
            self.beta_large - self.beta_small
        )

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])

    def get_loss(self, batch, batch_idx):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
        noise_imgs = []
        epsilons = torch.randn(batch.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        return loss
```

### 4.3 Training Step


```python
  def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer

```

### 4.4 Sampling

Generate new images using the trained model.

```python
    # Generate samples from denoising process
    gen_samples = []\n
    x = torch.randn((sample_batch_size, train_dataset.depth, train_dataset.size, train_dataset.size))\n
    sample_steps = torch.arange(model.t_range-1, 0, -1)\n
    for t in sample_steps:\n
        x = model.denoise_sample(x, t)\n
        if t % 50 == 0:\n
            gen_samples.append(x)\n
    for _ in range(n_hold_final):\n
        gen_samples.append(x)\n
    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)\n
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2
```

### 4.5 Run the Demo


* **Download the Dataset**:
   ```python
   import torch
   from torch.utils.data import Dataset
   from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
   from torchvision import transforms


   class DiffSet(Dataset):
       def __init__(self, train, dataset="MNIST"):
           transform = transforms.Compose([transforms.ToTensor()])

           datasets = {
               "MNIST": MNIST,
               "Fashion": FashionMNIST,
               "CIFAR": CIFAR10,
           }

           train_dataset = datasets[dataset](
               "./data", download=True, train=train, transform=transform
           )

           self.dataset_len = len(train_dataset.data)

           if dataset == "MNIST" or dataset == "Fashion":
               pad = transforms.Pad(2)
               data = pad(train_dataset.data)
               data = data.unsqueeze(3)
               self.depth = 1
               self.size = 32
           elif dataset == "CIFAR":
               data = torch.Tensor(train_dataset.data)
               self.depth = 3
               self.size = 32
           self.input_seq = ((data / 255.0) * 2.0) - 1.0
           self.input_seq = self.input_seq.moveaxis(3, 1)

       def __len__(self):
           return self.dataset_len

      def __getitem__(self, item):
           return self.input_seq[item]

   ```
* **Train the Model**: Use the [`entry.ipynb`](./entry.ipynb) notebook to train model and sample generated images. 



## 5. Discussion

### 5.1 Advantages
#### 5.1.1 High-Quality Generation
One of the standout features of diffusion models is their ability to generate incredibly high-quality images. If you've ever seen those super realistic AI-generated photos and wondered how they were made, diffusion models are often the magic behind them. They excel at capturing fine details and producing images that look almost indistinguishable from real ones. High-quality generation is crucial for applications where realism is key, such as in media and entertainment (think movies and video games), fashion, and even art. When the generated content needs to blend seamlessly with real-world data, diffusion models are a top choice.
#### 5.1.2 Stable Training
Training machine learning models can sometimes feel like trying to balance on a tightrope. Generative Adversarial Networks (GANs), for instance, are known for their tricky training process, which can be unstable and prone to issues like mode collapse (where the model only produces a limited variety of outputs).

Diffusion models, on the other hand, offer a much more stable training process. They don't rely on the adversarial setup of GANs (where two networks compete against each other), which means fewer headaches and more consistent results.
#### 5.1.3 Flexibility
Diffusion models aren't just a one-trick pony. They're highly flexible and can be adapted to generate a variety of data types beyond just images. From audio to text and even more complex data types, diffusion models can handle it all.

### 5.2 Limitations
#### 5.2.1 Computationally Intensive
One of the biggest hurdles you'll face with diffusion models is their demand for computational resources. Training these models involves a large number of steps to add and then remove noise from the data. This process can be quite slow and resource-intensive.
If you're working with limited computational power or have tight deadlines, diffusion models might not be the best choice. They require significant time and powerful hardware (like GPUs or TPUs) to train effectively. This can be a major barrier for smaller organizations or individual researchers.
#### 5.2.2 Complexity in Implementation
Diffusion models are not the simplest models to implement. The process of designing and tuning both the forward (noise addition) and reverse (denoising) steps can be quite complex. This includes carefully choosing the noise schedule and the number of timesteps.

#### 5.2.3 Slow Sampling Time
Generating new data samples with diffusion models is often slower compared to other generative models. Since the reverse process involves iteratively removing noise step-by-step, it can take a considerable amount of time to generate each sample.

In applications where real-time or near-real-time generation is crucial, such as interactive art or live data augmentation, the slow sampling time can be a significant limitation. This makes diffusion models less suitable for scenarios requiring quick outputs.

#### 5.2.4 Hyperparameter Sensitivity
Diffusion models can be quite sensitive to their hyperparameters, such as the number of timesteps, learning rates, and noise schedules. Finding the right set of hyperparameters often requires extensive experimentation and fine-tuning.

### 5.3 Comparison and Relationship to Other Methods
#### 5.3.1 GANs (Generative Adversarial Networks)
* **Similarities:** Both diffusion models and GANs aim to generate high-quality data samples.
* **Differences:** GANs use a discriminator to distinguish between real and generated data, while diffusion models rely on a forward and reverse process. GANs can suffer from mode collapse, where the generator produces limited varieties of samples, a problem less prevalent in diffusion models.
#### 5.3.2 VAEs (Variational Autoencoders)
* **Similarities:** Both are probabilistic models and involve learning a latent representation of the data.
* **Differences:** VAEs encode data into a latent space and decode it back, optimizing a variational lower bound, while diffusion models work directly in the data space with a series of noising and denoising steps.

### 5.4 Important Subsequent Representation Learning Algorithms
* **Score-Based Generative Models:** These models learn a score function (gradient of the data distribution) and use Langevin dynamics to generate samples. They share a conceptual similarity with diffusion models in using gradients for generation.
* **Denoising Diffusion Implicit Models (DDIM):** An extension of diffusion models that provides a more efficient sampling process, reducing the number of steps required to generate high-quality samples. DDIM introduceS a deterministic alternative to the stochastic nature of traditional diffusion models. Instead of relying on a stochastic reverse process, DDIM uses a non-Markovian deterministic process for sampling, which significantly reduces the number of steps needed for generation.
* **Variational Diffusion Models (VDM):** VDM combines principles from variational inference and diffusion models to create a more flexible and powerful generative model. VDM leverages a variational lower bound to optimize the model, improving both training stability and generation quality.
* **Diffusion Image Transformers (DIT):** DIT integrates the power of diffusion models with the versatility of transformers, creating a hybrid approach for image generation. DIT leverages the transformer architecture to model complex dependencies in image data, enhancing the generative capabilities of diffusion models.

### 5.5 Impactful Application Works
#### 5.5.1 OpenAI DALL-E
In January 2021, OpenAI released the text-to-image model DALL-E, its name being a play on surrealist artist Salvador Dali and the Pixar animated robot WALL-E. The model was based on a modified version of OpenAI’s remarkable GPT-3 text model, which had been released seven months before. DALL-E was a breakthrough in generative AI, demonstrating artistic abilities most people thought were impossible for a computer to possess.
The DALL-E model was not open sourced nor released to the public, but it inspired multiple researchers and hobbyists to attempt to replicate the research. The most popular of these models was DALL-E Mini, released in July 2021 (renamed Craiyon a year later at the request of OpenAI), and although it gained a cult following on social media, the quality was considerably poorer than the official DALL-E model. OpenAI published a paper announcing DALL-E 2 in April 2022 [[8]](#8), and the quality was significantly higher, attracting a waitlist of one million people.

Access was limited to waitlist users until September 2022, due to concerns about AI ethics and safety. Generation of images containing people was initially banned, as were a long list of sensitive words. Researchers identified DALL-E 2 adding the words black or female to some image prompts like a photo of a doctor in a hamfisted attempt to address bias inherited from the dataset (images of doctors on the internet are disproportionally of white males).

The team added inpainting and outpainting to the user interface in August 2022, which was a further leap forward, garnering attention in the press and on social media. These features allowed users to generate only selected parts of an image or to zoom out by generating around the border of an existing image. However, users have little control over the parameters of the model and could not fine-tune it on their own data. The model would generate garbled text on some images and struggled with realistic depictions of people, generating disfigured or deformed hands, feet, and eyes.

#### 5.5.2 Midjourney

In July 2022, just three months after the release of DALL-E 2, Midjourney put its v3 model in open beta. This was a uniquely good time to launch an image generation model, because the demonstrations of what DALL-E 2 could do from early users looked like magic, and yet access was initially limited. Eager early-adopters flocked to Midjourney, and its notable fantasy aesthetic gained a cult following among the gaming and digital art crowds, showcased in the **_now famous image_**, which won first prize in a digital art competition.

Midjourney was one of the first viable image models that had a business model and commercial license, making it suitable for more than just experimentation. The subscription model was favored by many artists accustomed to paying monthly for other software like Adobe Photoshop. It also helped the creative process to not be charged per image generated, particularly in the early days when you’d have to try multiple images before you found one that was high-enough quality. If you were a paying customer of Midjourney, you owned the rights to any image generated, unlike DALL-E, where OpenAI was retaining the copyright.

Unique to Midjourney is its heavy community focus. To use the tool, you must sign into a Discord server and submit your prompt in an open channel or direct message. Given that all image generations are shared in open channels by default, and private mode is available only on the most expensive plan, the vast majority of images created through Midjourney are available for others to learn from. This led to rapid copying and iteration between users, making it easy for novices to quickly learn from others. As early as July 2022, the Discord community was nearing 1 million people, and a year later, there were more than 13 million members.

When you find an image you like, you can click a button to upscale the image (make it higher resolution) for use. Many have speculated that this procedure acts as training data for reinforcement learning, similar to **_reinforcement learning from human feedback (RLHF)_**, the method touted as the key to success of ChatGPT. In addition, the team regularly asks for ratings of images generated by newer models in order to improve the performance. Midjourney released v4 of its model in November 2022, followed by v5 in March 2023 and v6 in December 2023.

## References

<a id="1">[1]</a> Foster, David. Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play. 2nd ed., O’Reilly Media, 2023.

<a id="2">[2]</a> Kulkarni, Akshay, et al. “Introduction to Generative AI.” Applied Generative AI for Beginners, Apress, 2023, pp. 1–13.

<a id="3">[3]</a> Sanseviero, Omar, et al. Hands-on Generative AI with Transformers and Diffusion Models. O’Reilly Media, 2024.

<a id="4">[4]</a> Jascha Sohl-Dickstein et al. "Deep Unsupervised Learning Using Nonequilibrium Thermodynamics." *arXiv preprint arXiv:1503.03585* (2015).

<a id="5">[5]</a> Yang Song and Stefano Ermon. "Generative Modeling by Estimating Gradients of the Data Distribution," *arXiv preprint arXiv:1907.05600* (2019).

<a id="6">[6]</a> Yang Song and Stefano Ermon, “Improved Techniques for Training Score-Based Generative Models,” *arXiv preprint arXiv:2006.09011* (2020).

<a id="7">[7]</a> Jonathon Ho et al. "Denoising Diffusion Probabilistic Models." *arXiv preprint arXiv:2006.11239* (2020).

<a id="8">[8]</a> Chitwan Saharia et al. "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding." *arXiv preprint arXiv:2205.11487* (2022).


## External Resources
* [The example of the first version of DALL-E’s capabilities.](https://openai.com/index/dall-e/)
* [The image created by Jason Michael Allen with the generative artificial intelligence platform Midjourney.](https://en.wikipedia.org/wiki/Th%C3%A9%C3%A2tre_D%27op%C3%A9ra_Spatial)
* [Reinforcement learning from human feedback (RLHF)](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback)

## Author Team

**Chenxingyu Huang**: Finish this whole final project.

