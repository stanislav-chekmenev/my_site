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

Proteins are essential biomolecules responsible for nearly every crucial process in living organisms. These macromolecules fold into complicated three-dimensional structures, determining how they interact with other molecules and shaping their biological roles. For instance, [hemoglobin](http://doi.org/10.2210/rcsb_pdb/mom_2003_5), a protein present in red blood cells, binds and transports oxygen from the lungs to tissues, which is a critical step in respiration {{<citenote 1>}}.

{{< sidebysideright src="/img/protein_discovery/hb-animation.gif" alt="Hemoglobin binding oxygen" caption="A change in shape occurs at a subunit of hemoglobin when it binds an oxygen molecule in turquoise, influencing the other subunits and increasing its binding affinity." >}}
Hemoglobin is composed of four subunits, each containing a [heme group](https://en.wikipedia.org/wiki/Heme) {{<citenote 2>}} that binds oxygen. This binding is a gradual and cooperative process: once the first heme binds oxygen, the structure of the whole protein is slightly changed, making it easier for the other heme groups to bind. Therefore, while binding the first molecule is relatively difficult, adding three consequent ones is progressively easier. There's plenty of oxygen in the lungs and it's easier for hemoglobin to bind it and quickly fill up the remaining subunits. However, as blood circulates through the rest of the body, where the carbon dioxide level is increased, a release of one of the bound oxygen molecules induces a change in the shape of the protein. This prompts the remaining three oxygens to be quickly released, as well. This cooperative mechanism allows hemoglobin to pick up large loads of oxygen from our lungs and deliver them where needed.
{{< /sidebysideright >}}

Hemoglobin's example demonstrates the sophisticated nature of protein structure and function relationships, refined over millions of years of evolution. Yet the task of intentional *de novo* protein design that aims to create entirely new protein structures with desirable functions from scratch is a complex and challenging one. Proteins consist of sequences of 20 standard amino acid residues. Therefore, if we limit the sequence length to, say, 50 residues &dash; even though one can find way longer sequences in nature &dash; the size of the possible design space is 20<sup>50</sup>. While traditional physics-based computational approaches, e.g. molecular dynamics (MD) simulations, could yield potentially promising results, high computation costs and slow speed significantly limit areas of their application. The design space is simply too vast for them to accomplish the task.

The rise of Bio ML and the phenomenal success of AlphaFold-2 marked a transformative moment in protein science. [AlphaFold-2](https://www.nature.com/articles/s41586-021-03819-2) {{<citenote 3>}} showed that deep learning could predict protein structures from amino acid sequences with high accuracy, outperforming all previous computational methods. This accelerated the adoption of machine learning techniques to the protein discovery problem. One of the latest outstanding results is a [model](https://www.nature.com/articles/s41586-023-05993-x) {{<citenote 4>}} that designed a protein binder for receptors of COVID-19.

More recently, generative models emerged as a powerful tool in protein design, capable of generating entirely new realistic protein sequences. Among these novel approaches, [FoldFlow-2](https://arxiv.org/abs/2405.20313) {{<citenote 5>}} caught my attention for a number of reasons. First of all, it leverages several cutting-edge methods that I wanted to learn because some are already used in many other protein discovery models and others offer significant improvements over current baselines. Flow matching on the SE(3) group manifold, Optimal Transport theory, sequence based protein representation modelling, and some AlphaFold-2 innovations like Invariant Point Attention define the FoldFlow-2 architecture. I was genuinely eager to dive into that knowledge. Secondly, despite the model being rather complex, it doesn't have an extremely large codebase or require weeks of GPU runtime to train. Considering all this, I chose FoldFlow-2, since I was also interested in modifying its architecture and experimenting with adding SE(3)-equivariant tensor field graph neural networks, which were on my study list too.

In the next chapters, I’ll unpack the essential machine learning innovations that underpin FoldFlow-2, share my experience of dissecting and familiarizing myself with its architecture, and detail my attempts at improving upon its already impressive capabilities.

## How FoldFlow-2 Fits Into Generative Protein Modelling

Although most people in the ML community are now familiar with the family of AlphaFold models and its revolutionary success in structure prediction, a new wave of research focuses on generative models that can design entirely new protein structures. Following that wave, FoldFlow-2 has been developed by the [Dreamfold](https://www.dreamfold.ai/) {{<citenote 6>}} team in Canada. It's a recent state-of-the-art $\text{SE}(3)^N$-invariant generative model for protein backbone generation that is additionally conditioned on the sequences of amino acids. As the name suggests, this architecture builds on top of [FoldFlow](https://arxiv.org/abs/2310.02391) ({{<citenote 7>}}) and implements a novel mechanism of handling multi-modal data, resulting in a substantial performance gain over the original version. 

Several successful attempts to create generative models ([RDM](https://dl.acm.org/doi/10.5555/3600270.3600469) {{<citenote 8>}}, [RFDiffusion](https://www.nature.com/articles/s41586-023-06415-8) {{<citenote 9>}}, [FrameDiff](https://arxiv.org/abs/2302.02277) {{<citenote 10>}}), operating on Riemannian manifolds, had been published before FoldFlow was released in 2024. Some required pretraining on prediction of protein structure (RFDiffusion), others used approximations to compute Riemannian divergence in the objective (RDM) and all of them relied on the Stochastic Differential Equations as the theoretical base for modelling the diffusion process on the manifold, which assumes a non-informative source (prior) distribution that one uses for training. FoldFlow was one of the first models that introduced SE(3) [Conditional Flow Matching](https://arxiv.org/abs/2210.02747) {{<citenote 11>}} for generation of a protein backbone with a possibility to use an informative prior distribution, and it utilized Riemannian [Optimal Transport](https://arxiv.org/abs/2302.00482) {{<citenote 12>}} to speed up training.

Generation of proteins from scratch is a much harder problem than predicting its 3D structure. A model should create proteins that are designable, different to the ones found in the training set and diverse. It's not only difficult to build such models, but it's also not easy to adequately assess their performance (more on this in the following sections). A multi-modal architecture of FoldFlow-2 is definitely a step forward that offers improvements across all three metrics that researchers use for evaluation. To fully grasp FoldFlow-2’s approach, let’s first cover some theoretical preliminaries and talk about the core ML techniques proposed by the authors of the paper.

## Overview of Core ML Techniques in FoldFlow-2

The model shares and extends some of the theoretical foundations laid out in the AlphaFold-2 and FrameDiff papers. Each of its techniques is a topic in itself and requires more detailed explanations than I can provide here without making this post excessively long. Instead, as I already mentioned in the beginning, I'll dive deeper into each technique in separate focused posts, offering a shorter description here. Let's kick off with an important concept of protein backbone and its parametrization.

###  Representations of a Protein Backbone

{{< sidebysideleft src="/img/protein_discovery/gly_ala_gram_schmidt.png" alt="Glycine and alanine amino acids" caption="Two amino acids are linked together. GLY - glycine, ALA - alanine. The vectors formed by two atomic bonds (shown with arrows) are used in the Gram-Schmidt algorithm to construct the initial frames for each residue. A torsion angle $\psi$ is required for correct oxygen placement." >}}
A backbone consists of repeated  N&mdash;C$\_{\alpha}$&mdash;C&mdash;O four heavy atoms linked together in a chain, with each set corresponding to one amino acid (residue). C$\_{\alpha}$ atom of each residue, except for glycine (GLY), is attached to a side chain that varies for each amino acid and determines its distinct chemical properties. The geometry of the backbone is determined by mapping a set of idealized local coordinates, [N$^{\star}$, C$^{\star}_{\alpha}$, C$^{\star}$, O$^{\star}$] $\in \mathbb{R^3}$ centered at <br> C$\_{\alpha}^{\star}$=(0, 0, 0), to the actual position of each residue. This mapping is performed using a rigid transformation given by an action $x$ of the <em>special Euclidean group</em> $\text{SE}(3)$ defined by 3D rotations $R$ and translations $S$. In other words, an action $x^i$ generates backbone coordinates for a residue $i$:
{{< /sidebysideleft >}}

<div id="eq:idealised2coords"></div>
$$[N, C_{\alpha}, C, O]^i = x^{i} \cdot [N^{\star}, C^{\star}_{\alpha}, C^{\star}, O^{\star}] \tag{1}$$ 

As shown in [Eq. 1](#eq:idealised2coords), each residue transformation can be decomposed into two components $x^i=(r^i, s^i)$ where $r^i \in \text{SO}(3)$ is a $3 \times 3$ rotation matrix and $s^i \in \mathbb{R^3}$ is a three-dimensional translation vector. Thus, following AlphaFold-2's approach, the entire structure of a protein with N residues is parameterized by a sequence of N such transformations described by the product group $\text{SE}(3)^N$. This results in a representation of all heavy atoms of the protein given by the tensor $X \in \mathbb{R}^{N \times 4 \times 3}$. Additionally, in order to compute the coordinates of the backbone oxygen in frame $i$, one needs to apply a rotation around C$\_{\alpha}$&mdash;C bond by a torsion angle $\psi^i$. 

The final rotation matrix $r^i$ for each residue is obtained via the Gram-Schmidt algorithm. This procedure operates on two vectors built from backbone atom coordinates, enforcing orthonormality to output a valid rotation matrix centered on the C$_{\alpha}$ atom. Further details of this parametrization are well documented in the appendix of the [FrameDiff](https://arxiv.org/abs/2302.02277) paper. 

So, one way to model a protein is to associate an element of $\text{SE}(3)$, called a "rigid" for simplicity, to each residue in the chain. This representation is used as the "structure" modality of the model. 

The second modality represents a protein as a sequence of 20 possible one-hot encoded amino acids. This is a usual way to tokenize data in protein language models. The whole protein sequence thus is provided by a tensor $A \in \mathbb{R}^{N \times 20}$.

Before moving on, let me briefly write what exactly the SE(3) group is and why it's the natural choice for describing protein structures.

### $\text{SE}(3)$ Group: Tool for Backbone Structure Parametrization 

Each rigid transformation $x^i$ corresponding to residue $i$ in a protein backbone is mathematically described by the [$\text{SE}(3)$ group](https://link.springer.com/book/10.1007/978-3-319-13467-3) {{<citenote 13>}}. Simply put, $\text{SE}(3)$ represents all possible rotations and translations in three-dimensional space. Since each residue's coordinates can be obtained according to [Eq. 1](#eq:idealised2coords), $\text{SE}(3)$ provides an ideal mathematical tool for modelling the spatial positions of amino acids. Essentially, our task is to predict the rotations and translations of the idealized coordinates for each residue, yielding the three-dimensional structure of the protein.

A powerful property of $\text{SE}(3)$ is that it forms a Lie group, which is also a differentiable manifold with smooth (differentiable) group operations. Informally, a manifold is a topological space that locally resembles Euclidean space. Since the manifold is differentiable in our case, we can smoothly interpolate between different points (representing different protein structures), which is crucial for generative modeling. Each point on this manifold has an associated [tangent space](https://en.wikipedia.org/wiki/Tangent_space){{<citenote 14>}}, allowing us to define smooth transitions, or *flows*, between protein structures. The tangent space at the identity element of $\text{SE}(3)$ is called its Lie algebra $\mathfrak{se}(3)$, which contains skew-symmetric rotation matrices and translation vectors. $\text{SE}(3)$ is a matrix Lie group, i.e. its elements are represented with matrices. Additionally, $\text{SE}(3)$ can be decomposed into two simpler groups: the rotation group $\text{SO}(3)$ and the translation group $\mathbb{R}^3$.

Next, I'll discuss how this group formalism enables the creation of flows on the manifold, which form the core of the protein generative process in FoldFlow-2.

### Conditional Flow Matching on the $\text{SE}(3)$ Manifold

In the previous subsection, I mentioned that Lie groups are smooth manifolds, a property that allows them to be equipped with a [Riemannian metric](https://en.wikipedia.org/wiki/Riemannian_manifold#Definition) {{<citenote 15>}}, which makes it possible to define distances, angles and [geodesics](https://en.wikipedia.org/wiki/Geodesic){{<citenote 16>}} on the manifold. For $\text{SE}(3)$, the metric decomposes into separate metrics for its constituent subgroups: $\text{SO}(3)$ and $\mathbb{R}^3$ {{<citenote 7>}}. The decomposition of the $\text{SE}(3)$ group into its subgroups allows to construct independent flows for rotations and translations, which can then be combined to create a unified flow on $\text{SE}(3)$. Flow matching {{<citenote 11>}} techniques for Euclidean spaces like $\mathbb{R}^3$ are well-studied and you can find an excellent introduction in this [post](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html){{<citenote 17>}}. Thus, I'll focus on the key aspects of flow matching specifically for the rotation group $\text{SO}(3)$, with detailed coverage in upcoming posts. 


#### Metrics and Distances on $\text{SE}(3)$

Before diving into the flow matching framework, let me establish the notion of a metric and derive the distance for the $\text{SE}(3)$ group, which we can conveniently split into two components, $\text{SO}(3)$ and $\mathbb{R}^3$. A canonical choice for the metric on $\text{SO}(3)$ is:

<div id="eq:so3-metric"></div>
$$\langle \mathfrak{r_1}, \mathfrak{r_2} \rangle_\text{SO}(3) = \frac{1}{2} \text{tr}(\mathfrak{r_1}^T, \mathfrak{r_2}) \tag{2},$$

where $\mathfrak{r_1}$ and $\mathfrak{r_2}$ are the elements of the Lie algebra $\mathfrak{so}(3)$.

Using [eq. 3](#eq:so3-metric), the distance on $\text{SE}(3)$ can be defined as follows:

<div id="eq:se3-dist"></div>
$$d_{\text{SE}(3)}(x_1, x_2) = \sqrt{d_{\text{SO}(3)}(r_1, r_2)^2 + d_{\mathbb{R}^3}(s_1, s_2)^2} = \sqrt{\left\| \text{log}(r_1^T r_2) \right\|_F^2 + d_{\mathbb{R}^3}(s_1, s_2)^2} \tag{3}$$

Hence, the distance on $\text{SO}(3)$ is calculated as the Frobenius matrix norm of the logarithmic map (read [section 4.3.3](#sec:cfm)) of the relative rotation between $r_1$ and $r_2$ and $d_{\mathbb{R}^3}$ is the usual Euclidean distance.

This distance formulation will be crucial when I discuss the optimization objective and Optimal Transport in the following sections.

#### Probability Path on $\text{SO}(3)$

Another concept I must introduce before revealing the optimization objective of the model is probability paths on $\text{SO}(3)$. Imagine that we have two probability densities $\rho_0$, $\rho_1 \in \text{SO}(3)$ where $\rho_0$ corresponds to our target data distribution and $\rho_1$ is an easy-to-sample source distribution. We can smoothly interpolate between these two densities in the probability space by following a *probability path* denoted $\rho_t: [0, 1] \to \mathbb{P}(\text{SO}(3))$ that depends on one parameter $t$ that can be perceived as time. This transition is generated by a *flow*, a mapping $\psi_t$ that takes every starting point $r$ in $\rho_0$, given by a rotation matrix on $\text{SO}(3)$, and moves it to a new location on the manifold, $r_t = \psi_t(r)$, at time $t$. The entire distribution $\rho_t$ is thus formed by applying this map to the initial distribution $\rho_0$. 

The map $\psi_t$ is the solution to the ordinary differential equation (ODE), [eq. 4](#eq:ode), with the initial condition $\psi_0(r) = r$.

<div id="eq:ode"></div>
$$\frac{d \psi_t}{dt} = u_t(\psi_t(r)) \tag{4}$$
​
The dynamics of this flow are governed by a velocity field, $u_t: [0,1] \times \text{SO}(3) \to T_{r_t}\text{SO}(3)$, that lies in the tangent space $T_{r_t}\text{SO}(3)$ at point $r_t = \psi_t(r)$. This means the velocity field $u_t$ assigns a tangent vector to each point on the manifold. Therefore, for any rotation $r$, the velocity $u_t(r) \in T_{r_t}$ is a vector in the tangent space at that point, describing the instantaneous direction and magnitude of the flow.

Now that we understand how probability paths and flows work on the $\text{SO}(3)$ manifold, let's see how FoldFlow-2 leverages these concepts to formulate its training objective. 

<div id="sec:cfm"></div>

#### From Conditional Flow Matching to the Optimization Objective

The main task of the model is to generate realistic and novel proteins, which are parametrized by the product group $\text{SE}(3)^N$. One way to train such a model is by using the Conditional Flow Matching {{<citenote 11>}} technique. Focusing on the rotation ($\text{SO}(3)$) component of the objective, let me shed some light upon the main idea of this approach.

The idea of conditional flow matching is to fit a conditional velocity field $u_t(r_t| r_0, r_1)$ in the tangent space $T_{r_t}\text{SO}(3)$ associated with the flow $\psi_t$ that smoothly transports the data distribution $r_0 \sim \rho_0$ to the source distribution $r_1 \sim \rho_1$. The unconditional vector field, the marginal velocity field over all possible endpoint pairs, is intractable to compute directly. Therefore, the model learns a conditional vector field $u_t(r_t| r_0, r_1)$, which is conditioned on the specific start ($r_0$) and end ($r_1$) points of the trajectory. Once we have access to this vector field, we can sample from $\rho_1$ and use a simple ODE solver to run the reverse process, generating a protein that resembles those found in the data distribution $\rho_0$. This generative technique can trace its roots back to the influential paper on [Neural ODEs](https://arxiv.org/abs/1806.07366){{<citenote 18>}} where the authors introduced continuous normalizing flows. I recommend reading it to those unfamiliar with the topic, since it provides foundational concepts that simplify the understanding of flow matching.

The authors of FoldFlow-2 follow the most natural strategy developed for the previous version of the model{{<citenote 7>}} that constructs a flow $\psi_t$, connecting $r_0$ and $r_1$, by utilizing the geodesic between these points. A geodesic that connects two points is the shortest path between them on a manifold. For a general manifold, including $\text{SO}(3)$, the geodesic interpolant between $r_0$ and $r_1$, indexed by $t$, is given by the following equation:

<div id="eq:geo-interpolant"></div>
$$\psi_t = r_t = \exp_{r_0} (t \, \log_{r_0}(r_1)) \tag{5}$$

A geodesic interpolant, [eq. 5](#eq:geo-interpolant) between two points $r_0$ and $r_1$ on a manifold is the generalization of linear interpolation to curved spaces. In Euclidean space, interpolating between two points is simply a straight line. However, on manifolds, straight lines are generalized by geodesics, which are curves that locally minimize distance and have zero acceleration. This interpolant includes two concepts important for manifold operations: *exponential* and *logarithmic* maps. 

The exponential map ([fig. 3a](#fig:exp-log-maps)) $\exp_{r_0}: T_{r_0}\text{SO}(3) \to \text{SO}(3)$ takes a tangent vector $v \in T_{r_0}\text{SO}(3)$ at point $r_0$ and transports it along the unique geodesic $\gamma$, which satisfies $\gamma(0) = r_0$ and $\dot{\gamma}(0) = v$, in the direction specified by that vector, producing a new rotation matrix:

<div id="eq:exp-map"></div>
$$r_1 = \exp_{r_0}(v) = \gamma(1) \in \text{SO}(3) \tag{6}$$  

Effectively, one travels a unit of time along $\gamma$ and ends up at a new point on the manifold. It can be viewed as an analogue to addition in Euclidean space: for a point $p$ and a tangent vector $v$ the exponential map is simply $\exp_p(v) = p + v$.

Conversely, the logarithmic map ([fig. 3b](#fig:exp-log-maps)) is the inverse operation $\log_{r_0}: \text{SO}(3) \to T_{r_0}\text{SO}(3)$. It provides the tangent vector at $r_0$ pointing in the direction of $r_1$:

<div id="eq:log-map"></div>
$$v = \log_{r_0}(r_1) \in T_{r_0}\text{SO}(3) \tag{7} $$

In Euclidean space, it is analogous to vector subtraction between two points: for points $p$ and $q$, the log map  $\log_p(q)$ returns the vector $q-p$.

{{< centerimage src="/img/protein_discovery/exp_log_maps.svg" alt="Exponential and logarithmic maps" caption="(a) Exponential map between $r_0$ and $r_1$. (b) Logarithmic map between $r_0$ and $r_1$." id="fig:exp-log-maps" >}}

One of the key innovations of the paper is an efficient computation of the logarithmic and exponential maps required for the geodesic interpolant [eq. 5](#eq:geo-interpolant), avoiding their standard definitions as infinite matrix series. This method leverages the Lie group structure of SO(3). To compute $\log_{r_0}(r_1)$, a relative rotation $r_{rel} = r_1^T r_0$ is first calculated. Then $r_{rel}$ is converted to its [axis-angle representation](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation){{<citenote 19>}} and the *hat operator* is applied to it. The hat operator maps a three-dimensional vector to a skew-symmetric matrix. Since the output of the previous step is a skew-symmetric matrix, this whole procedure yields $\mathfrak{r_1} \in \mathfrak{so}(3)$ that belongs to the Lie algebra of $\text{SO}(3)$ and, by definition, lives at the tangent space at the identity element of the group. It is possible to apply *left translation* to $\mathfrak{r_1}$ to move it to the tangent space of $r_0$. This is achieved, using left matrix multiplication by $r_0$, which produces the desired logarithmic map $\log_{r_0}(r_1)$. Similarly, the exponential map can be computed in closed form for skew-symmetric matrices, the elements of $\mathfrak{so}(3)$, using Rodrigues' formula. You can see the visualization of the steps involved in the log map computation in [fig. 4](#fig:compute-log-map).

{{< centerimage src="/img/protein_discovery/compute_log_map.svg" alt="Visualization of the computation of the logarithmic map $\log_{r_0}(r_1)$" caption="(a) Two rotations on $\text{SO}(3)$ group: $r_0$ and $r_1$. (b) Relative rotation $r_{rel}$ that, by definition, has its starting point at the identity element $Id$ located at the center of the sphere. (c) The log map is efficiently computed in the axis-angle representation at the identity element, yielding a tangent vector of the Lie algebra, $\log_{Id}(r_{rel}) \in \mathfrak{so}(3)$. (d) The tangent vector is left-translated to $r_0$, producing $\log_{r_0}(r_1)$." id="fig:compute-log-map" >}}

Having established feasible ways to work with the geodesic interpolant $r_t$, the authors describe how to get the conditional velocity field $u_t(r_t| r_0, r_1)$, which is given by the ODE associated with the conditional flow and, in our case, is the time derivative of the geodesic interpolant [eq. 5](#eq:geo-interpolant):

<div id="eq:cond-vec-field-ode"></div>
$$\frac{d\psi_t(r|r_0, r_1)}{dt} = \dot{r_t} = u_t(r_t| r_0, r_1) \tag{8}$$

The computation of the conditional vector field leverages the group structure rather than directly taking the derivative of the interpolant. From [eq. 8](#eq:cond-vec-field-ode), we see that computing the vector field $u_t \in T_{r_t}\text{SO}(3)$ requires taking the time derivative of $r_t$ at time $t$ along the geodesic. However, the vector field has a simple closed-form expression written in the paper as $u_t = \log_{r_t}(r_0) / t$, where the logarithmic map is computed using the efficient procedure described above. In my opinion, there's a minus sign missing in that formula, but it doesn't really matter for training because our neural network learns to "undo" that mistake adjusting its weights accordingly.

Finally, I'm ready to write down the full loss of FoldFlow-2, where the translational part is derived similarly to the rotational component. The readers can check the suggested  [post](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html){{<citenote 17>}} for further details.

<div id="eq:loss-so3"></div>
$$ \mathcal{L} = \mathcal{L}_{\text{SO}(3)} + \mathcal{L}_{\mathbb{R}^3} = \mathbb{E}_{t \sim \mathcal{U}[0,1], q(x_0, x_1), \rho_t(x_t|x_0, x_1, \bar{a})} \left\| v_\theta(t, r_t, \bar{a}) - \log_{r_t}(r_0) / t \right\|^2_{\text{SO}(3)} + \left\| v_\theta(t, s_t, \bar{a}) - \frac{s_t - s_0}{t} \right\| ^2_2 \tag{9}$$

where $t$ is sampled uniformly from $\mathcal{U}[0,1]$, $v_{\theta} \in T_{r_t}\text{SO}(3)$ is in the tangent space at $r_t$, the norm is induced by the metric on $\text{SO}(3)$ (I'll cover it in the chapter about implementation details) and $q(r_0, r_1)$ is any coupling between samples from the data and prior distributions. An optimal choice, developed in the paper, is to set $q(r_0, r_1) = \pi(r_0, r_1)$, which is a solution of the Riemannian Optimal Transport{{<citenote 7>}} problem. Let me say a couple of words about this last important aspect.

 
### Optimal Transport on $\text{SE}(3)$

While it's possible to use any coupling between the data, $\rho_0$, and the source, $\rho_1$, distributions, e.g. $q(\rho_0, \rho_1) = \rho_0 \rho_1$, it's not guaranteed that the probability path, generated by the conditional vector field $u_t(x_t|x_0, x_1)$, would be the shortest when measured under an appropriate metric on $\text{SE}(3)$. Shorter, more optimal paths are desirable, as they lead to faster, more stable training and decreasing variance in the training objective {{<citenote 12>}}. To achieve this, FoldFlow-2 uses a mathematical approach called Optimal Transport (OT). There's a great introductory lecture on [Optimal Transport](https://www.youtube.com/watch?v=k1CeOJdQQrc&ab_channel=MLinPL){{<citenote 20>}} recorded two years ago, which would be very beneficial for deeper understanding of the subject. A python package [POT](https://pythonot.github.io/quickstart.html){{<citenote 21>}} provides an excellent starting point into OT, as well.

The goal of OT, according to [transportation theory](https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)){{<citenote 22>}}, is to find the most efficient way to "transport" one distribution to another, minimizing the total effort. For example, to learn how much iron ore to ship from each mine to each factory so that every factory gets exactly what it needs, every mine sends out all its iron ore, and the total shipping cost is as small as possible. That was an important problem during the World War II.

Formally, Optimal Transport finds the best "transport map" $\Psi$ that minimizes the overall cost of moving all the points. This is captured by the following formula:

<div id="eq:ot-plan"></div>
$$\text{OT}(\rho_0, \rho_1) = \underset{\Psi: \Psi_{\text{#} \rho_0=\rho_1} }{\text{inf}} \int_{\text{SE}(3)^N_0} \frac{1}{2} c(x, \Psi(x))^2 d\rho_0(x) \tag{10}$$

{{< sidebysideright src="/img/protein_discovery/monge_map.svg" alt="Optimal Transport map" caption="Optimal Transport map for two continuous distributions $\rho_0$ and $\rho_1$." >}}
In other words, [eq. 10](#eq:ot-plan) defines the minimum possible cost, denoted $\text{OT}(\rho_0, \rho_1)$, to transform an entire data distribution of frames, corresponding to the protein residues, $\rho_0$, into the distribution $\rho_1$, following the best possible transport map, $\Psi$. It's a *pushforward map*. The notation $\Psi\_{\text{#}}\rho_0$ represents the new distribution formed by applying the map $\Psi$ to all points in the original distribution $\rho_0$. The constraint, $\Psi\_{\text{#}}\rho_0 = \rho_1$, dictates that this new, transformed distribution must be identical to the target distribution $\rho_1$. It guarantees that we reshape the entire "cloud" of starting points into the target cloud, rather than just moving points arbitrarily ([fig. 5](#fig:ot-monge-map)). The cost for moving a single point $x$ to its destination $\Psi(x)$ is determined by the cost function $c(x, \Psi(x))$, which is computed according to the geodesic distance ([eq. 3](#eq:se3-dist)) induced by the metric on $\text{SE}(3)$. The cost formulation heavily penalizes long, inefficient paths and strongly encourages the model to find the most direct transformation route on the manifold. 
{{< /sidebysideright >}}

Searching for the perfect transport map $\Psi$ for thousands of points is computationally very difficult. The paper employs a well-known shortcut. Instead of solving for $\Psi$ directly, it solves a related, more manageable problem (the Kantorovich formulation of OT) to find an optimal transport plan $\pi$, which is a joint probability distribution minimizing the cost of transporting $\rho_0$ to $\rho_1$. This plan does not define the full map but provides an efficient way to sample corresponding pairs of points $x_0$ and $x_1$. By training on these matched pairs, $(x_0, x_1)$ sampled from $\pi$, FoldFlow-2 ensures a much more efficient and stable learning process. The visual intuition of this pair sampling is presented in [Fig. 6](#fig:ot-plan-sampling) 

{{< centerimage src="/img/protein_discovery/ot_sampling.svg" alt="OT plan sampling" caption="Sampling the points from $\rho_0$ and $\rho_1$, using the Optimal Transport plan that minimizes the transportation cost induced by the metric on $\text{SE}(3)$. (a) Sampling without OT. (b) Sampling according to the Optimal Transport plan." id="fig:ot-plan-sampling" >}}

### Summary of the Core ML Techniques of FoldFlow-2

In this chapter, we explored the sophisticated machine learning framework that powers FoldFlow-2's ability to generate protein structures. The process begins by representing protein backbones as a sequence of rigid transformations on the special Euclidean group, $\text{SE}(3)$. This geometric foundation is crucial, as it allows the model to operate on a smooth manifold where distances and paths are well-defined.

The core generative mechanism is Conditional Flow Matching{{<citenote 11>}}, a technique where the model learns a time-dependent velocity field. This field defines a flow that smoothly transports points from the complex data distribution of valid protein structures to the easy-to-sample source distribution. 

To further enhance training stability and speed, FoldFlow-2 incorporates Riemannian Optimal Transport{{<citenote 7>}} (OT). Instead of arbitrarily pairing initial and target structures, OT is used to find the most efficient coupling, or "transport plan," between the two distributions. By training on these optimally matched pairs, the model learns straighter, more direct transformation paths, which reduces the variance of the training objective and leads to a more robust learning process. 

With this theoretical foundation in place, I will now examine how these concepts are implemented in FoldFlow-2's actual architecture.

## Overview of the Model Architecture

The main innovation of FoldFlow-2 in comparison to the original version is the addition of a powerful sequence encoder. At a high level, FoldFlow-2 consists of three main stages that follow a typical Encoder-Processor-Decoder ([fig. 7](#fig:architecture)) deep learning paradigm:
 1. Input structure and sequence are passed to the encoder.
 2. Encoded representations are combined and processed in a multi-modal trunk.
 3. Processed representations are sent to the geometric decoder, which outputs a vector field that lies on the tangent space of SE(3) group.

{{< centerimage src="/img/protein_discovery/foldflow2_architecture.svg" alt="FoldFlow-2 architecture" caption="FoldFlow-2 architecture." id="fig:architecture" >}}

### Structure & Sequence Encoder

Structure encoding is performed with a module based on the Invariant Point Attention (IPA) and the protein backbone update algorithms that have been designed for AlphaFold-2. IPA modifies the standard [attention mechanism](https://arxiv.org/abs/1706.03762) {{<citenote 23>}} by making the attention weights depend on distances between key and query *points* that are two sets of N three-dimensional points where N is a hyperparameter. These points are obtained through a linear projection layer applied to residue features, similar to how standard keys and queries are produced. The baked-in invariance of IPA and the way the backbone is updated make the module SE(3)-equivariant. You can find more details about the algorithms in the [supplementary material](https://www.nature.com/articles/s41586-021-03819-2#Sec20).{{<citenote 24>}} The block's output is divided into three types of representations that follow the naming convention of AlphaFold-2: *single*, *pair* and *rigid*. Without going too deep into what those embeddings are, I'd like to point out that single representations are essentially transformed residue features, pair representations are computed for each pair of residues, using their features and relative distances, and rigids are elements of SE(3) group I briefly introduced above that describe each residue in terms of rotations and translations.

{{< sidebysideright src="/img/protein_discovery/sequence_to_trunk.svg" alt="Glycine and alanine amino acids" caption="Information flow in the sequence-to-trunk module." id="fig:seq2trunk" >}}
The core component of the sequence encoder is a pre-trained frozen <a href="https://www.science.org/doi/10.1126/science.ade2574">ESM-2</a>{{<citenote 25>}} model with 650M parameters. This protein language model was trained on masked sequences of amino acids and creates high-quality features with strong generalization properties, making them well-suited for downstream tasks. The model extracts embeddings from each transformer layer and the final prediction head, yielding 34 total 1280-dimensional feature vectors per residue as single representations. Additionally, attention weights between all pairs of residues from each layer are stacked together to form the final pair representations. The sequence-to-trunk module then constructs a learned linear combination of those 34 embeddings (called "Learnable Pooling" in the figure) before transforming the result with an MLP. Meanwhile, the pair representations are projected into a lower-dimensional space via an MLP and combined with embedded pairwise distances ([fig. 8](#fig:seq2trunk)). 
{{< /sidebysideright >}}

### Multi-Modal Fusion Trunk

Both modalities are mixed and processed in the multi-modal fusion trunk that consists of two main parts: the combiner module and the trunk blocks. 

{{< sidebysideleft src="/img/protein_discovery/combiner_module.svg" alt="Combiner module" caption="Information flow in the combiner module." id="fig:combiner">}}
The combiner ([fig. 9](#fig:combiner)) uses dedicated MLPs to project each type of single and pair embedding into a shared latent space with half the original dimensionality. The resulting feature vectors from sequence and structure encodings are then concatenated to create unified single and pair joint representations. These are fed further to the trunk module that is made up of 2 Triangular Self Attention blocks, which used as the core units of the Evoformer block in AlphaFold-2. Therefore, the whole component is a compact version of the Evoformer architecture with additional shallow MLP mixing of the input embeddings of two different modalities. 
{{< /sidebysideleft >}} 

### Geometric Decoder

Finally, the structure decoder leverages the IPA transformer once more and decodes its input into $\text{SE}(3)_0^N$ vector fields. $\text{SE}(3)_0^N$ is a translation-invariant version of $\text{SE}(3)^N$ that is constructed by switching to a reference frame centered at the center of mass of all C<sub>&alpha;</sub> backbone atoms. This module takes as input the single and pair embeddings from the trunk, along with the rigids from the structure encoder. The authors found that adding a skip-connection between the decoder and encoder was crucial for model performance, since it preserved temporal information, which would otherwise be lost within the Evoformer block.

### Model Summary

To wrap up this chapter, let me summarize the key aspects of FoldFlow-2 that I've covered so far:

- It works directly on the $\text{SE}(3)^N_0$ manifold and it's $\text{SE}(3)^N$-invariant.
- Multi-modality is supported via fusing sequence and structure representations.
- Many of its componets are inspired by the original AlphaFold-2 algorithms.


{{< references >}}
<li id="ref-1">Goodsell, Dutta, <a href="http://doi.org/10.2210/rcsb_pdb/mom_2003_5">Molecule of the month</a>, 2003. <a href="javascript:goBackToLastCitation('1')">↩</a></li>
<li id="ref-2"><a href="https://en.wikipedia.org/wiki/Heme">Heme group</a>, Wikipedia. <a href="javascript:goBackToLastCitation('2')">↩</a></li>
<li id="ref-3">Jumper et al., <a href="https://www.nature.com/articles/s41586-021-03819-2">Highly accurate protein structure prediction with AlphaFold</a>. Nature, 2021 <a href="javascript:goBackToLastCitation('3')">↩</a></li>
<li id="ref-4">Gainza et. al., <a href="https://www.nature.com/articles/s41586-023-05993-x">De novo design of protein interactions with learned surface fingerprints</a>. Nature, 2023 <a href="javascript:goBackToLastCitation('4')">↩</a></li>
<li id="ref-5">Huguet et. al., <a href="https://arxiv.org/abs/2405.20313">Sequence-augmented SE(3)-flow matching for conditional protein backbone generation</a>. NeurIPS, 2024 <a href="javascript:goBackToLastCitation('5')">↩</a></li>
<li id="ref-6"><a href="https://www.dreamfold.ai/">Dreamfold</a>. <a href="javascript:goBackToLastCitation('6')">↩</a></li>
<li id="ref-7">Bose et. al., <a href="https://arxiv.org/abs/2310.02391">SE(3)-Stochastic flow matching for protein backbone generation</a>. ICLR, 2024 <a href="javascript:goBackToLastCitation('7')">↩</a></li>
<li id="ref-8">Huang et. al., <a href="https://dl.acm.org/doi/10.5555/3600270.3600469">Riemannian diffusion models</a>. NIPS, 2022 <a href="javascript:goBackToLastCitation('8')">↩</a></li>
<li id="ref-9">Watson et. al., <a href="https://www.nature.com/articles/s41586-023-06415-8">De novo design of protein structure and function with RFdiffusion</a>. Nature, 2023 <a href="javascript:goBackToLastCitation('9')">↩</a></li>
<li id="ref-10">Yim et. al., <a href="https://arxiv.org/abs/2302.02277">SE(3) diffusion model with application to protein backbone generation</a>. PMLR, 2023 <a href="javascript:goBackToLastCitation('10')">↩</a></li>
<li id="ref-11">Lipman et. al., <a href="https://arxiv.org/abs/2210.02747">Flow matching for generative modeling</a>. ICLR, 2023 <a href="javascript:goBackToLastCitation('11')">↩</a></li>
<li id="ref-12">Tong et. al., <a href="https://arxiv.org/abs/2302.00482">Improving and generalizing flow-based generative models with minibatch optimal transport</a>. TMLR, 2024 <a href="javascript:goBackToLastCitation('12')">↩</a></li>
<li id="ref-13">Hall, <a href="https://link.springer.com/book/10.1007/978-3-319-13467-3">Lie groups, Lie algebras, and representations</a>. Springfield, 2013 <a href="javascript:goBackToLastCitation('13')">↩</a></li>
<li id="ref-14"><a href="https://en.wikipedia.org/wiki/Tangent_space">Tangent space</a>, Wikipedia. <a href="javascript:goBackToLastCitation('14')">↩</a></li>
<li id="ref-15"><a href="https://en.wikipedia.org/wiki/Riemannian_manifold#Definition">Riemannian manifold</a>, Wikipedia. <a href="javascript:goBackToLastCitation('15')">↩</a></li>
<li id="ref-16"><a href="https://en.wikipedia.org/wiki/Geodesic">Geodesic</a>, Wikipedia. <a href="javascript:goBackToLastCitation('16')">↩</a></li>
<li id="ref-17">Fjelde, Mathieu, Dutordoir, <a href="https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html">An Introduction to Flow Matching</a>. 2024 <a href="javascript:goBackToLastCitation('17')">↩</a></li>
<li id="ref-18">Chen et. al., <a href="https://arxiv.org/abs/1806.07366">Neural Ordinary Differential Equations</a>. NIPS, 2018 <a href="javascript:goBackToLastCitation('18')">↩</a></li>
<li id="ref-19"><a href="https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation">Axis-angle representation</a>, Wikipedia. <a href="javascript:goBackToLastCitation('19')">↩</a></li>
<li id="ref-20"><a href="https://www.youtube.com/watch?v=k1CeOJdQQrc&ab_channel=MLinPL">A primer on optimal transport theory and algorithms</a>, Youtube. <a href="javascript:goBackToLastCitation('20')">↩</a></li>
<li id="ref-21"><a href="https://pythonot.github.io/quickstart.html">POT package</a>. <a href="javascript:goBackToLastCitation('21')">↩</a></li>
<li id="ref-22"><a href="https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)">Transportation theory</a>, Wikipedia. <a href="javascript:goBackToLastCitation('22')">↩</a></li>
<li id="ref-23">Vaswani et. al., <a href="https://arxiv.org/abs/1706.03762">Attention is all you need.</a>. NIPS, 2017 <a href="javascript:goBackToLastCitation('23')">↩</a></li>
<li id="ref-24">Jumper et. al., <a href="https://www.nature.com/articles/s41586-021-03819-2#Sec20">Supplementary information for AlphaFold-2</a>. Nature, 2021 <a href="javascript:goBackToLastCitation('24')">↩</a></li>
<li id="ref-25">Lin et. al., <a href="https://www.science.org/doi/10.1126/science.ade2574">Evolutionary-scale prediction of atomic-level protein structure with a language model</a>. Science, 2023 <a href="javascript:goBackToLastCitation('25')">↩</a></li>
{{< /references >}}





