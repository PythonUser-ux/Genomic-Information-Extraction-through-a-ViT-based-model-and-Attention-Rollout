# Genomic-Information-Extraction-through-a-ViT-based-model-and-Attention-Rollout

# Abstract

Nowadays the implementation of Natural Language Processing ap-
plications - such as translators, document summarization or even
biological sequence analysis - has become strictly related to the
well-know Transformer deep learning model architecture. This
design has been repeatedly confirmed as the standard approach,
which should be always kept in mind when dealing with a certain
class of tasks; despite its huge potential in the aforementioned cases
the applications of Transformers is not to be thought only limited
to NLP as it may be exploited in the domain of Computer Vision as
well.
Indeed a quite recent paper titled "An image is worth 16x16 words"
further investigates the versatility of the architecture by propos-
ing the Vision Transformer model. When trained on large datasets
and then transferred to multiple average image recognition bench-
marks (ImageNet, CIFAR-100, VTAB, etc.), this model is capable
of surpassingly dealing with image recognition and also handling
the competition against the leading edge convolutional neural net-
works. Another merit of the Vision Transformer is that of being
essentially cheap to train. Although a Transformer does not exploit
the typically used image-specific inductive biases it should be noted
that it might implement its own ways to take them into account.
What we propose here is a model designed for medical images clas-
sification which is made by two parts: a Vision Transformer and a
convolutional classificator.
