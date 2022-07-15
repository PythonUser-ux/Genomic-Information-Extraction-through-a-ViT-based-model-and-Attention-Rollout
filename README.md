# Genomic-Information-Extraction-through-a-ViT-based-model-and-Attention-Rollout

[comment]: <> (I am currently dedicated to this thesis work: give a look to My_Model.ipynb to see what I have implemented so far.<br>)
[comment]: <> (I still need to train the Vision Transformer in order to generate the attention maps needed for the training of the full model.)

# Abstract

Nowadays the implementation of Natural Language Processing applications - such as translators, document summarization or even
biological sequence analysis - has become strictly related to the
well-know Transformer deep learning model architecture. <br> This
design has been repeatedly confirmed as the standard approach,
which should be always kept in mind when dealing with a certain
class of tasks; despite its huge potential in the aforementioned cases
the applications of Transformers is not to be thought only limited
to NLP as it may be exploited in the domain of Computer Vision as
well. <br>
Indeed a quite recent paper titled "An image is worth 16x16 words"
further investigates the versatility of the architecture by proposing the Vision Transformer model.<br> When trained on large datasets
and then transferred to multiple average image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), this model is capable
of surpassingly dealing with image recognition and also handling
the competition against the leading edge convolutional neural networks. Another merit of the Vision Transformer is that of being
essentially cheap to train. <br> Although a Transformer does not exploit
the typically used image-specific inductive biases it should be noted
that it might implement its own ways to take them into account.
What we propose here is a model designed for medical images classification which is made by two parts: a Vision Transformer and a
convolutional classificator. <br>

# The proposed model <br>

The model we propose here is a model made by two
parts. The first part is a Vision Transformer based model that will
be used for the generation of attention maps through the so-called
mechanism of attention rollout.<br> The second part is a 2D classificator that takes as input the very same concatenation of patches used
in the previous part but this time the input is concatenated with
the attention maps generated for that input by the ViT. The dataset
we use is the CT lung, and the patches feed into the ViT are a set
of 9 consecutive slices of a CT scan arranged in a 3x3 grid.<br> The
aim of the model is to classify the scans containing a tumor mass
depending on whether they are of genomic nature (binary classification).
