<p align="center">
  <a href="" rel="noopener">
 <img width=200px width=400px src="https://pl-flash-data.s3.amazonaws.com/assets_lightning/docs/images/logos/lightning-ai.png" alt="Project logo"></a>
</p>

<h3 align="center">This is learning Pytorch LightingAI Tutorials Repository Contains variety of Example of official Repository.
</h3>

<div align="center">

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)](https://pypi.org/project/pytorch-lightning/)
[![PyPI Status](https://badge.fury.io/py/pytorch-lightning.svg)](https://badge.fury.io/py/pytorch-lightning)
[![PyPI Status](https://pepy.tech/badge/pytorch-lightning)](https://pepy.tech/project/pytorch-lightning)
[![Conda](https://img.shields.io/conda/v/conda-forge/pytorch-lightning?label=conda&color=success)](https://anaconda.org/conda-forge/pytorch-lightning)
[![DockerHub](https://img.shields.io/docker/pulls/pytorchlightning/pytorch_lightning.svg)](https://hub.docker.com/r/pytorchlightning/pytorch_lightning)
[![codecov](https://codecov.io/gh/Lightning-AI/lightning/branch/master/graph/badge.svg?token=SmzX8mnKlA)](https://codecov.io/gh/Lightning-AI/lightning)

</div>

---

<p align="center"> Few lines describing your project.
    <br> 
</p>


## 📝 Table of Contents

- [About](#about)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Tutorials](../TODO.md)
- [Built Using](#built_using)
- [Authors](#authors)
- [References](#acknowledgement)

## 🧐 About <a name = "about"></a>

PyTorch Lightning is an open-source Python library that provides a high-level interface for PyTorch, a popular deep learning framework.[1] It is a lightweight and high-performance framework that organizes PyTorch code to decouple the research from the engineering, making deep learning experiments easier to read and reproduce. It is designed to create scalable deep learning models that can easily run on distributed hardware while keeping the models hardware agnostic.

In 2019, Lightning was adopted by the NeurIPS Reproducibility Challenge as a standard for submitting PyTorch code to the conference.[2]

In 2022, the PyTorch Lightning library officially became a part of the Lightning framework, an open-source framework managed by the original creators of PyTorch Lightning. 

## 🏁 Installation <a name = "installation"></a>
Simple installation from PyPI

```bash
pip install pytorch-lightning
```

<!-- following section will be skipped from PyPI description -->

<details>
  <summary>Other installation options</summary>
    <!-- following section will be skipped from PyPI description -->

#### Install with optional dependencies

```bash
pip install pytorch-lightning['extra']
```

#### Conda

```bash
conda install pytorch-lightning -c conda-forge
```

#### Install stable 1.7.x

The actual status of 1.7 \[stable\] is the following:

[![Test PyTorch full](https://github.com/Lightning-AI/lightning/actions/workflows/ci-pytorch-test-full.yml/badge.svg?branch=release%2Fpytorch&event=push)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-pytorch-test-full.yml?query=branch%3Arelease%2Fpytorch)
[![Test PyTorch with Conda](https://github.com/Lightning-AI/lightning/actions/workflows/ci-pytorch-test-conda.yml/badge.svg?branch=release%2Fpytorch&event=push)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-pytorch-test-conda.yml?query=branch%3Arelease%2Fpytorch)
[![TPU tests](https://dl.circleci.com/status-badge/img/gh/Lightning-AI/lightning/tree/release%2Fpytorch.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/Lightning-AI/lightning/tree/release%2Fpytorch)
[![Check Docs](https://github.com/Lightning-AI/lightning/actions/workflows/docs-checks.yml/badge.svg?branch=release%2Fpytorch&event=push)](https://github.com/Lightning-AI/lightning/actions/workflows/docs-checks.yml?query=branch%3Arelease%2Fpytorch)

Install future release from the source

```bash
pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/release/pytorch.zip -U
```

#### Install bleeding-edge - future 1.6

Install nightly from the source (no guarantees)

```bash
pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip -U
```

or from testing PyPI

```bash
pip install -iU https://test.pypi.org/simple/ pytorch-lightning
```

</details>
<!-- end skipping PyPI description -->


## ⛏️ Built Using <a name = "built_using"></a>

- [Pytorch](https://pytorch.org/) - Deep learning Library Design by [Facebook](https://github.com/facebook)
- [Python](https://www.python.org/) - Programming Language 

## ⛹️‍♂️ Tutorials
###### Hello world

- [ ] [MNIST hello world](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html)

###### Contrastive Learning

- [ ] [BYOL](https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/self_supervised.html#byol)
- [ ] [CPC v2](https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/self_supervised.html#cpc-v2)
- [ ] [Moco v2](https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/self_supervised.html#moco-v2-api)
- [ ] [SIMCLR](https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/self_supervised.html#simclr)

###### NLP

- [ ] [GPT-2](https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/convolutional.html#gpt-2)
- [ ] [BERT](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/text-transformers.html)

###### Reinforcement Learning

- [ ] [DQN](https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/reinforce_learn.html#dqn-models)
- [ ] [Dueling-DQN](https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/reinforce_learn.html#dueling-dqn)
- [ ] [Reinforce](https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/reinforce_learn.html#reinforce)

###### Vision

- [GAN](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/basic-gan.html)

###### Classic ML

- [Logistic Regression](https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/classic_ml.html#logistic-regression)
- [Linear Regression](https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/classic_ml.html#linear-regression)


## 🧑‍💻 Initial Start <a name="getting-started"></a>
### Step 1: Add these imports

```python
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
```

### Step 2: Define a LightningModule (nn.Module subclass)

A LightningModule defines a full *system* (ie: a GAN, autoencoder, BERT or a simple Image Classifier).

```python
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

**Note: Training_step defines the training loop. Forward defines how the LightningModule behaves during inference/prediction.**

### Step 3: Train!

```python
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

autoencoder = LitAutoEncoder()
trainer = pl.Trainer()
trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
```

## Advanced features

Lightning has over [40+ advanced features](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags) designed for professional AI research at scale.

Here are some examples:

<div align="center">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/features_2.jpg" max-height="600px">
</div>

<details>
  <summary>Highlighted feature code snippets</summary>

```python
# 8 GPUs
# no code changes needed
trainer = Trainer(max_epochs=1, accelerator="gpu", devices=8)

# 256 GPUs
trainer = Trainer(max_epochs=1, accelerator="gpu", devices=8, num_nodes=32)
```

<summary>Train on TPUs without code changes</summary>

```python
# no code changes needed
trainer = Trainer(accelerator="tpu", devices=8)
```

<summary>16-bit precision</summary>

```python
# no code changes needed
trainer = Trainer(precision=16)
```

<summary>Experiment managers</summary>

```python
from pytorch_lightning import loggers

# tensorboard
trainer = Trainer(logger=TensorBoardLogger("logs/"))

# weights and biases
trainer = Trainer(logger=loggers.WandbLogger())

# comet
trainer = Trainer(logger=loggers.CometLogger())

# mlflow
trainer = Trainer(logger=loggers.MLFlowLogger())

# neptune
trainer = Trainer(logger=loggers.NeptuneLogger())

# ... and dozens more
```

<summary>EarlyStopping</summary>

```python
es = EarlyStopping(monitor="val_loss")
trainer = Trainer(callbacks=[es])
```

<summary>Checkpointing</summary>

```python
checkpointing = ModelCheckpoint(monitor="val_loss")
trainer = Trainer(callbacks=[checkpointing])
```

<summary>Export to torchscript (JIT) (production use)</summary>

```python
# torchscript
autoencoder = LitAutoEncoder()
torch.jit.save(autoencoder.to_torchscript(), "model.pt")
```

<summary>Export to ONNX (production use)</summary>

```python
# onnx
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmpfile:
    autoencoder = LitAutoEncoder()
    input_sample = torch.randn((1, 64))
    autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
    os.path.isfile(tmpfile.name)
```

</details>

### Pro-level control of training loops (advanced users)

For complex/professional level work, you have optional full control of the training loop and optimizers.

```python
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # access your optimizers with use_pl_optimizer=False. Default is True
        opt_a, opt_b = self.optimizers(use_pl_optimizer=True)

        loss_a = ...
        self.manual_backward(loss_a, opt_a)
        opt_a.step()
        opt_a.zero_grad()

        loss_b = ...
        self.manual_backward(loss_b, opt_b, retain_graph=True)
        self.manual_backward(loss_b, opt_b)
        opt_b.step()
        opt_b.zero_grad()
```

______________________________________________________________________

## Advantages over unstructured PyTorch

- Models become hardware agnostic
- Code is clear to read because engineering code is abstracted away
- Easier to reproduce
- Make fewer mistakes because lightning handles the tricky engineering
- Keeps all the flexibility (LightningModules are still PyTorch modules), but removes a ton of boilerplate
- Lightning has dozens of integrations with popular machine learning tools.
- [Tested rigorously with every new PR](https://github.com/Lightning-AI/lightning/tree/master/tests). We test every combination of PyTorch and Python supported versions, every OS, multi GPUs and even TPUs.
- Minimal running speed overhead (about 300 ms per epoch compared with pure PyTorch).

______________________________________________________________________

## ✍️ Authors <a name = "authors"></a>

- [@Ashish Patel](https://github.com/ashishpatel26) - Idea & Initial work

See also the list of [contributors](https://github.com/kylelobo/The-Documentation-Compendium/contributors) who participated in this project.

## 🎉 References <a name = "acknowledgement"></a>
 1. "GitHub - PyTorch Lightning". 2019-12-01.
 2. "Reproducibility Challenge @NeurIPS 2019". NeurIPS. 2019-12-01. Retrieved 2019-12-01.
 3. https://github.com/Lightning-AI/lightning