+++
title = 'Diving Into Protein Design with SE(3) Flow Matching'
date = 2025-07-01T16:54:46+02:00
mermaid = false
math = true
draft = true
+++

# Transitioning to Bio ML: <br> My Experience Learning and Modifying FoldFlow-2

{{< toc >}}

## Introduction: Personal Motivation & the Road Ahead

Transition phases, especially those that shape the direction of our lives, are often among the most challenging yet deeply rewarding episodes we encounter. Currently being in such a phase, I'm nearing its logical conclusion by documenting my experience here. 

It's been almost a year since I decided to shift my career path towards Bio ML and the fascinating field of protein drug discovery in particular. Why proteins? &mdash; you might ask. With a background in particle physics, a thirst for solving complex puzzles, and expertise in various machine learning areas, I've always been driven by the desire to engage in work that is meaningful, intellectually demanding, and beneficial to humanity. Protein design offered precisely that combination. Setting my goal to enter the field, I embarked on a learning journey that turned out to be the most engaging self-initiated research project I've undertaken so far.

This post is the culmination of my endeavour and brings together all the pieces that play an important role in learning the latest generative techniques for proteins, with a hands-on deep dive into FoldFlow-2, a recent state-of-the-art model in protein structure generation. I’ll walk you through the advanced ML concepts underlying the model, document my process of understanding and modifying its architecture, and reflect on the insights and skills gained along the way. 

If you're curious about how modern ML approaches enable the creation of new proteins, or if you're considering redirecting your career similarly to mine, I hope this article provides useful perspectives and practical resources. I'd be happy if it serves as a springboard for a smoother start to your own journey into the fascinating world of Bio ML. For those who want to dig deeper, I’ll provide links to upcoming focused posts where I break down each core technique powering an augmented version of FoldFlow-2. So, stay tuned, the links will be added gradually as I continue writing.

## The Promise and Challenge of Protein Drug Discovery

Proteins are essential biomolecules responsible for nearly every crucial process in living organisms. These macromolecules fold into complicated three-dimensional structures, determining how they interact with other molecules and shaping their biological roles. For instance, [hemoglobin](http://doi.org/10.2210/rcsb_pdb/mom_2003_5), a protein present in red blood cells, binds and transports oxygen from the lungs to tissues, which is a critical step in respiration {{< citenote 1 >}}.

{{< sidebysideright src="/img/protein_discovery/hb-animation.gif" alt="Hemoglobin binding oxygen" caption="A change in shape occurs at a subunit of hemoglobin when it binds an oxygen molecule in turquoise, influencing the other subunits and increasing its binding affinity" >}}
Hemoglobin is composed of four subunits, each containing a [heme group](https://en.wikipedia.org/wiki/Heme) {{< citenote 2 >}} that binds oxygen. This binding is a gradual and cooperative process: once the first heme binds oxygen, the structure of the whole protein is slightly changed, making it easier for the other heme groups to bind. Therefore, while binding the first molecule is relatively difficult, adding three consequent ones is progressively easier. There's plenty of oxygen in the lungs and it's easier for hemoglobin to bind it and quickly fill up the remaining subunits. However, as blood circulates through the rest of the body, where the carbon dioxide level is increased, a release of one of the bound oxygen molecules induces a change in the shape of the protein. This prompts the remaining three oxygens to be quickly released, as well. This cooperative mechanism allows hemoglobin to pick up large loads of oxygen from our lungs and deliver them where needed.
{{< /sidebysideright >}}

Hemoglobin's example demonstrates the sophisticated nature of protein structure and function relationships, refined over millions of years of evolution. Yet the task of intentional *de novo* protein design that aims to create entirely new protein structures with desirable functions from scratch is a complex and challenging one. Proteins consist of sequences of 20 standard amino acid residues. Therefore, if we limit the sequence length to, say, 50 residues &dash; even though one can find way longer sequences in nature &dash; the size of the possible design space is 20<sup>50</sup>. While traditional physics-based computational approaches, e.g. molecular dynamics (MD) simulations, could yield potentially promising results, high computation costs and slow speed significantly limit areas of their application. The design space is simply too vast for them to accomplish the task.

The rise of Bio ML and the phenomenal success of AlphaFold-2 marked a transformative moment in protein science. [AlphaFold-2](https://www.nature.com/articles/s41586-021-03819-2) {{< citenote 3 >}} showed that deep learning could predict protein structures from amino acid sequences with high accuracy, outperforming all previous computational methods. This accelerated the adoption of machine learning techniques to the protein discovery problem. One of the latest outstanding results is a [model](https://www.nature.com/articles/s41586-023-05993-x) {{< citenote 4 >}} that designed a protein binder for receptors of COVID-19.

More recently, generative models emerged as a powerful tool in protein design, capable of generating entirely new realistic protein sequences. Among these novel approaches, [FoldFlow-2](https://arxiv.org/abs/2405.20313) {{< citenote 5 >}} caught my attention for a number of reasons. First of all, it leverages several cutting-edge methods that I wanted to learn because some are already used in many other protein discovery models and others offer significant improvements over current baselines. Flow Matching on the SE(3) group manifold, Optimal Transport theory, sequence based protein representation modelling, and some AlphaFold-2 innovations like Invariant Point Attention define the FoldFlow-2 architecture. I was genuinely eager to dive into that knowledge. Secondly, despite the model being rather complex, it doesn't have an extremely large codebase or require weeks of GPU runtime to train. Considering all this, I chose FoldFlow-2, since I was also interested in modifying its architecture and experimenting with adding SE(3)-equivariant tensor field graph neural networks, which were on my study list too.

In the next chapters, I’ll unpack the essential machine learning innovations that underpin FoldFlow-2, share my experience of dissecting and familiarising myself with its architecture, and detail my attempts at improving upon its already impressive capabilities.

## How FoldFlow-2 Fits Into Generative Protein Modelling

Although most people in the ML community are now familiar with the family of AlphaFold models and its revolutionary success in structure prediction, a new wave of research focuses on generative models that can design entirely new protein structures. Following that wave, FoldFlow-2 has been developed by the [Dreamfold](https://www.dreamfold.ai/) {{< citenote 6 >}} team in Canada. It's a recent state-of-the-art $\text{SE}(3)^N$-invariant generative model for protein backbone generation that is additionally conditioned on the sequences of amino acids. As the name suggests, this architecture builds on top of [FoldFlow](https://arxiv.org/abs/2310.02391) ({{< citenote 7 >}}) and implements a novel mechanism of handling multi-modal data, resulting in a substantial performance gain over the original version. 

Several successful attempts to create generative models ([RDM](https://dl.acm.org/doi/10.5555/3600270.3600469) {{< citenote 8 >}}, [RFDiffusion](https://www.nature.com/articles/s41586-023-06415-8) {{< citenote 9 >}}, [FrameDiff](https://arxiv.org/abs/2302.02277) {{< citenote 10 >}}), operating on Riemannian manifolds, had been published before FoldFlow was released in 2024. Some required pretraining on prediction of protein structure (RFDiffusion), others used approximations to compute Riemannian divergence in the objective (RDM) and all of them relied on the Stochastic Differential Equations as the theoretical base for modelling the diffusion process on the manifold, which assumes a non-informative prior distribution that one uses for training. FoldFlow was one of the first models that introduced SE(3) [Conditional Flow Matching](https://arxiv.org/abs/2210.02747) {{< citenote 11 >}} for generation of a protein backbone with a possibility to use an informative prior distribution, and it utilized Riemannian [Optimal Transport](https://arxiv.org/abs/2302.00482) {{< citenote 12 >}} to speed up training.

Generation of proteins from scratch is a much harder problem than predicting its 3D structure. A model should create proteins that are designable, different to the ones found in the training set and diverse. It's not only difficult to build such models, but it's also not easy to adequately assess their performance (more on this in the following sections). A multi-modal architecture of FoldFlow-2 is definitely a step forward that offers improvements across all three metrics that researchers use for evaluation. To fully grasp FoldFlow-2’s approach, let’s first cover some theoretical preliminaries and talk about the core ML techniques proposed by the authors of the paper.

## Overview of Core ML Techniques in FoldFlow-2

The model shares and extends some of the theoretical foundations laid out in the AlphaFold-2 and FrameDiff papers. Each of its techniques is a topic in itself and requires more detailed explanations than I can provide here without making this post excessively long. Instead, as I already mentioned in the beginning, I'll dive deeper into each technique in separate focused posts, offering a shorter description here. Let's kick off with an important concept of protein backbone and its parametrization.

###  Representations of a Protein Backbone

{{< sidebysideleft src="/img/protein_discovery/gly_ala_gram_schmidt.png" alt="Glycine and alanine amino acids" caption="Two amino acids are linked together. GLY - glycine, ALA - alanine. The vectors formed by two atomic bonds (shown with arrows) are used in the Gram-Schmidt algorithm to construct the initial frames for each residue. A torsion angle $\psi$ is required for correct oxygen placement." >}}
A backbone consists of repeated  N&mdash;C$\_{\alpha}$&mdash;C&mdash;O four heavy atoms linked together in a chain, with each set corresponding to one amino acid (residue). C$\_{\alpha}$ atom of each residue, except for glycine (GLY), is attached to a side chain that varies for each amino acid and determines its distinct chemical properties. The geometry of the backbone is determined by mapping a set of idealised local coordinates, [N$^{\star}$, C$^{\star}_{\alpha}$, C$^{\star}$, O$^{\star}$] $\in \mathbb{R^3}$ centered at <br> C$\_{\alpha}^{\star}$=(0, 0, 0), to the actual position of each residue. This mapping is performed using a rigid transformation given by an action $x$ of the <em>special Euclidean group</em> $\text{SE}(3)$ defined by 3D rotations $R$ and translations $T$. In other words, an action $x^i$ generates backbone coordinates for a residue $i$:
{{< /sidebysideleft >}}

$$[N, C_{\alpha}, C, O]^i = x^{i} \cdot [N^{\star}, C^{\star}\_{\alpha}, C^{\star}, O^{\star}]$$

Each transformation $x^i$ can be decomposed into two components $x^i=(r^i, t^i)$ where $r^i \in \text{SO}(3)$ is a $3 \times 3$ rotation matrix and $t^i \in \mathbb{R^3}$ is a three-dimensional translation vector. Thus, following AlphaFold-2's approach, the entire structure of a protein with N residues is parameterized by a sequence of N such transformations described by the product group $\text{SE}(3)^N$. This results in a representation of all heavy atoms of the protein given by the tensor $X \in \mathbb{R}^{N \times 4 \times 3}$. Additionally, in order to compute the coordinates of the backbone oxygen in frame $i$, one needs to apply a rotation around C$\_{\alpha}$&mdash;C bond by a torsion angle $\psi^i$. 

The final rotation matrix $r^i$ for each residue is obtained via the Gram-Schmidt algorithm. This procedure operates on two vectors built from backbone atom coordinates, enforcing orthonormality to output a valid rotation matrix centered on the C$_{\alpha}$ atom. Further details of this parametrization are well documented in the appendix of the [FrameDiff](https://arxiv.org/abs/2302.02277) paper. 

So, one way to model a protein is to associate an element of $\text{SE}(3)$, called a "rigid" for simplicity, to each residue in the chain. This representation is used as the "structure" modality of the model. 

The second modality represents a protein as a sequence of 20 possible one-hot encoded amino acids. This is a usual way to tokenize data in protein language models. The whole protein sequence thus is provided by a tensor $A \in \mathbb{R}^{N \times 20}$.

## Overview of the Model Architecture

The main innovation of FoldFlow-2 in comparison to the original version is the addition of a powerful sequence encoder. At a high level, FoldFlow-2 consists of three main stages that follow a typical Encoder-Processor-Decoder deep learning paradigm:
 1. Input structure and sequence are passed to the encoder.
 2. Encoded representations are combined and processed in a multi-modal trunk.
 3. Processed representations are sent to the geometric decoder, which outputs a vector field that lies on the tangent space of SE(3) group.

{{< centerimage src="/img/protein_discovery/foldflow2_architecture.svg" alt="FoldFlow-2 architecture" caption="FoldFlow-2 architecture" >}}

### Structure & Sequence Encoder

Structure encoding is performed with a module based on the Invariant Point Attention (IPA) and the protein backbone update algorithms that have been designed for AlphaFold-2. IPA modifies the standard [attention mechanism](https://arxiv.org/abs/1706.03762) {{< citenote 13 >}} by making the attention weights depend on distances between key and query *points* that are two sets of N three-dimensional points where N is a hyperparameter. These points are obtained through a linear projection layer applied to residue features, similar to how standard keys and queries are produced. The baked-in invariance of IPA and the way the backbone is updated make the module SE(3)-equivariant. You can find more details about the algorithms in the [supplementary material](https://www.nature.com/articles/s41586-021-03819-2#Sec20).{{< citenote 14 >}} The block's output is divided into three types of representations that follow the naming convention of AlphaFold-2: *single*, *pair* and *rigid*. Without going too deep into what those embeddings are, I'd like to point out that single representations are essentially transformed residue features, pair representations are computed for each pair of residues, using their features and relative distances, and rigids are elements of SE(3) group I briefly introduced above that describe each residue in terms of rotations and translations.

{{< sidebysideright src="/img/protein_discovery/sequence_to_trunk.svg" alt="Glycine and alanine amino acids" caption="Information flow in the sequence-to-trunk module." >}}
The core component of the sequence encoder is a pre-trained frozen <a href="https://www.science.org/doi/10.1126/science.ade2574">ESM-2</a>{{< citenote 15 >}} model with 650M parameters. This protein language model was trained on masked sequences of amino acids and creates high-quality features with strong generalization properties, making them well-suited for downstream tasks. The model extracts embeddings from each transformer layer and the final prediction head, yielding 34 total 1280-dimensional feature vectors per residue as single representations. Additionally, attention weights between all pairs of residues from each layer are stacked together to form the final pair representations. The sequence-to-trunk module then constructs a learned linear combination of those 34 embeddings (called "Learnable Pooling" in the figure) before transforming the result with an MLP. Meanwhile, the pair representations are projected into a lower-dimensional space via an MLP and combined with embedded pairwise distances (Fig. 4). 
{{< /sidebysideright >}}

### Multi-Modal Fusion Trunk

Both modalities are mixed and processed in the multi-modal fusion trunk that consists of two main parts: the combiner module and the trunk blocks. 

{{< sidebysideleft src="/img/protein_discovery/combiner_module.svg" alt="Combiner module" caption="Information flow in the combiner module." >}}
The combiner uses dedicated MLPs to project each type of single and pair embedding into a shared latent space with half the original dimensionality. The resulting feature vectors from sequence and structure encodings are then concatenated to create unified single and pair joint representations. These are fed further to the trunk module that is made up of 2 Triangular Self Attention blocks, which used as the core units of the Evoformer block in AlphaFold-2. Therefore, the whole component is a compact version of the Evoformer architecture with additional shallow MLP mixing of the input embeddings of two different modalities (Fig 5.). 
{{< /sidebysideleft >}} 

### Geometric Decoder

Finally, the structure decoder leverages the IPA transformer once more and decodes its input into $\text{SE}(3)_0^N$ vector fields. $\text{SE}(3)_0^N$ is a translation-invariant version of $\text{SE}(3)^N$ that is constructed by switching to a reference frame centered at the center of mass of all C<sub>&alpha;</sub> backbone atoms. This module takes as input the single and pair embeddings from the trunk, along with the rigids from the structure encoder. The authors found that adding a skip-connection between the decoder and encoder was crucial for model performance, since it preserved temporal information, which would otherwise be lost within the Evoformer block.

### Model Summary

To wrap up this chapter, let me summarise the key aspects of FoldFlow-2 that I've covered so far:

- It works directly on the $\text{SE}(3)^N_0$ manifold and it's $\text{SE}(3)^N$-invariant.
- Multi-modality is supported via fusing sequence and structure representations.
- Many of its componets are inspired by the original AlphaFold-2 algorithms.


{{< references >}}
<li id="ref-1">Goodsell, Dutta, <a href="http://doi.org/10.2210/rcsb_pdb/mom_2003_5">Molecule of the month</a>, 2003. <a href="#cite-1">↩</a>
<li id="ref-2"><a href="https://en.wikipedia.org/wiki/Heme">Heme group</a>, Wikipedia. <a href="#cite-2">↩</a>
<li id="ref-3">Jumper et al., <a href="https://www.nature.com/articles/s41586-021-03819-2">Highly accurate protein structure prediction with AlphaFold</a>. Nature, 2021 <a href="#cite-3">↩</a>
<li id="ref-4">Gainza et. al., <a href="https://www.nature.com/articles/s41586-023-05993-x">De novo design of protein interactions with learned surface fingerprints</a>. Nature, 2023 <a href="#cite-4">↩</a>
<li id="ref-5">Huguet et. al., <a href="https://arxiv.org/abs/2405.20313">Sequence-augmented SE(3)-flow matching for conditional protein backbone generation</a>. NeurIPS, 2024 <a href="#cite-5">↩</a>
<li id="ref-6"><a href="https://www.dreamfold.ai/">Dreamfold</a>. <a href="#cite-6">↩</a>
<li id="ref-7">Bose et. al., <a href="https://arxiv.org/abs/2310.02391">SE(3)-Stochastic flow matching for protein backbone generation
</a>. ICLR, 2024 <a href="#cite-7">↩</a>
<li id="ref-8">Huang et. al., <a href="https://dl.acm.org/doi/10.5555/3600270.3600469">Riemannian diffusion models</a>. NIPS, 2022 <a href="#cite-8">↩</a>
<li id="ref-9">Watson et. al., <a href="https://www.nature.com/articles/s41586-023-06415-8">De novo design of protein structure and function with RFdiffusion</a>. Nature, 2023 <a href="#cite-9">↩</a>
<li id="ref-10">Yim et. al., <a href="https://arxiv.org/abs/2302.02277">SE(3) diffusion model with application to protein backbone generation
</a>. PMLR, 2023 <a href="#cite-10">↩</a>
<li id="ref-11">Lipman et. al., <a href="https://arxiv.org/abs/2210.02747">Flow matching for generative modeling
</a>. ICLR, 2023 <a href="#cite-11">↩</a>
<li id="ref-12">Tong et. al., <a href="https://arxiv.org/abs/2302.00482">Improving and generalizing flow-based generative models with minibatch optimal transport
</a>. TMLR, 2024 <a href="#cite-12">↩</a>
<li id="ref-13">Vaswani et. al., <a href="https://arxiv.org/abs/1706.03762">Attention is all you need.
</a>. NIPS, 2017 <a href="#cite-13">↩</a>
<li id="ref-14">Jumper et. al., <a href="https://www.nature.com/articles/s41586-021-03819-2#Sec20">Supplementary information for AlphaFold-2
</a>. Nature, 2021 <a href="#cite-14">↩</a>
<li id="ref-15">Lin et. al., <a href="https://www.science.org/doi/10.1126/science.ade2574">Evolutionary-scale prediction of atomic-level protein structure with a language model</a>. Science, 2023 <a href="#cite-15">↩</a>
</li>
{{< /references >}}





