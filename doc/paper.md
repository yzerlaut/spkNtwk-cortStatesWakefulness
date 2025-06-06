# Adaptable spiking dynamics enables flexible computations across waking states in sensory cortex

# Spiking Dynamics of State-Dependent Thalamo-Cortical Processing during Wakefulness

## Abstract

Despite the large amount of experimental evidence highlighting the critical role of state-dependent computations in the awake sensory cortex, a theoretical framework providing a mechanistic and functional description of this phenomenon is still lacking. The present study builds a minimal spiking network model of a local cortical assembly and examines its dynamical and computational properties (and compares ... with neurophysiological recordings in the awake behaving mouse). It is first shown that a weak intrinsic rhythmicity coupled with competing inhibitory and disinhibitory circuits allow recurrent spiking networks to generate the set of dynamical regimes observed at various arousal levels in rodent sensory cortices (also known as the “U-model” of cortical states). We then test the functional specificity of such network states in two vision-inspired tasks. In a natural scene discrimination task, the network encoding is found to be optimal (highly-reliable) when displaying the sparse activity regime associated to intermediate arousal levels. In a low contrast detection task, the network is optimally detecting (high-sensitivity) when displaying the dense activity regime found at high arousal levels. Finally, the rhythmic activity characterizing the idling cortex is shown to insure unambiguous transitions between the high-reliability and high-sensitivity modes of cortical activity, thus giving it a a functional role. This theoretical model therefore demonstrates a key function of spiking networks: they can implement an unambiguous switch between high-reliability and high-sensitivity modes. 

This feature might be one of the defining principle of sensory perception during active 

 underlying the context-dependent modulation of perception is to implement an adaptive strategy to reliably switch between high-reliability and high-sensitivity modes and therefore .

## Main Text
  
### Intro

The variable nature of information transmission in neuronal networks is a puzzling aspect of the computation performed by sensory cortices. Indeed, unlike artificial intelligent systems that extract information by fixed transformations of incoming signals, the cortical representations of sensory features are highly variable across repeated presentations of a given stimulus (REFS). 

A striking feature of information processing in the sensory cortex of awake animals 

Over the last three decades, in vivo physiology has clearly established that the modulation of signal processing by the internal brain dynamics is a hallmark of cortical function (Arieli et al., 1996).

Since the early/seminal evidence that the ongoing dynamics of neural activity was critically shaping cortical computations [1], much effort has been devoted to the understanding of the functional role of such a phenomenon. Recent research in awake rodents has highlighted the correlation between behavioral and physiological states and specific network regimes in neocortex [2,3]. Such findings thus led to the idea that the regimes of cortical activity are the substrate underlying the functional flexibility (Cardin, 2019; McGinley et al., 2015b).
In this paper, we tackle this question with theoretical modeling of spiking network dynamics to focus our investigation on spike-based computation.

Recent investigation in the cortex of awake rodents have made this 
In parallel, psychophysical studies have pinpointed the critical role of the behavioral and physiological state in controlling perception.  
The circuits and mechanisms have been the fo

### Dynamics - the U-model of arousal

The U-model of arousal is defined by a $V_m$ pattern in pyramidal cells as this quantitiy has been evidenced as a robust experimental signature of cortical states (McGinley et al., 2015b). 

[[Figure {dynamics} around here]]

### Dynamics - setting up the model
We first investigate what are the minimal biophysical and circuit ingredients to reproduce the set of cortical regimes during wakefulness as captured by the U-model of arousal. Following an emerging consensus (McGinley et al., 2015a), we define the U-model of cortical states by its $V_m$ signature in pyramidal cells, (Figure {dynamics}a)


We embed a subpopulation of layer 5 excitatory neurons with oscillatory ionic currents, giving them the ability to display the rhythmic spiking , similarly to findings under anesthesia. 
For the rhythmic part, we adapt

To derive a general dynamical picture and prevent the emergent dynamics to depend on specific assymetries of biophysical parameters, we set all cellular and synaptic parameters identical. The only degree of freedom shaping individual population dynamics in this minimal model of the cortical assembly is the connectivity 

To capture the effective strength of each circuit components with a single parameter, we set all cellular and synaptic properties as equal and the only degree of freedom of the model is the 7x5 connectivity matrix (). 

The equation for the neuron i was:
\begin{equation}
C_m \frac{dV}{dt} =\frac{1}{Rm} \, (V_m-E_L) + I_{syn}(t) 
\end{equation}
The membrane parameters were $E_L$=-75mv, $C_m$=200pF, Rm =100MΩ, Vthre=-50mV and Vreset=EL, and Isyn(t) is the total synaptic currents targeting a neuron. The synaptic 

### Dynamics - investigation

Following evidences of subcortical control (Reimer et al., 2014a), I hypothesize a single variable controlling arousal-mediated, capturing cholinergic input
we investigate the dynamics as function of this variable

Fig 1. 

To explore the network dynamics in this 49-dimensional space (7x7 connectivity probabilities), we make use of the analytical reduction of the network dynamics as numerical investigations is too computationnally expensive (intractable/prohibitove/unfeasible). 
investigate network dynamics at the time scale of network state modulation (>1s)

gaining 3 orders of magnitude in computation time for a single simulation (from $\sim$10s to $\sim$0.1s)

The wide gamma envelope ([20,80]Hz) in the mean-field is approximated by computing the fraction of the $V_m$ variance due to the inhibitory synaptic events targetting the pyramidal population. 

Such an analytical reduction of the network dynamics, also known as a mean-field description, to perform the connectivity parameter search. Those analytical estimates were confirmed by numerical simulations of the spiking network, in particular the optimal configuration was found 

### Dynamics - Results

Remarkably we found that the connnectivity schemes allowing the closely resemble the canonical scheme of cortical circuit (Jiang et al., 2015), we therefore identify (see Figure {1})

The largest inhibitory population was found to highly connect to pyramidal cell, to balance the recurrent dynamics at high-activity level (see current weight inset in Figure {dynamics}e(ii)), i.e. matching the PV basket-cell type of inhibition.

The second largest population was providing tonic inhibition during rhythmic periods and decreased

Finally, it should be noted that a 5 population model containing the three stereotypical inhibitory populations (PV, SST, VIP) could display the same behavior with .

### Function - introducing the visual task

### Role of rhythmic activity

At that point, two crucial questions emerge. What is the functional role of the non-zero endogenous activity found at low arousal ? Such rhythmic pattern make it difficult to be optimal for the transmission of sensory-evoked activity. 
Given the ability of a cortical network to adapt its functional mode, there is the need for a downstream network to decode such mode so that the readout process can be modulated as well. How does that decoding works and what insures it's reliability ?
 one might hypo . I therefore tested the ability of the resting state pattern to disambiguate the decoding of a 
Unambiguous state switch 
This was implemented by considering alternative scenari to the "U-model" of cortical states (found for other configuration, see Figure {1}). 
 network model 


allows an unambiguous decoding of 
it serves as an ongoing to disambiguate the switch to function-specific sparse and dense activity desynchronized regimes are either a globally more inhibited or a globally more activated state with respect to the rhythmic pattern,  it allows an unambiguous and reliable decoding of network state modulation by a recipient network. 

Because the two function-specific desynchronized regimes are either a globally more inhibited or a globally more activated state with respect to the rhythmic pattern,  it allows an unambiguous and reliable decoding of network state modulation by a recipient network.

Fig 4. A spiking network model for the cortical states of wakefulness: architecture and dynamical regimes. (a) jkfhsakdfh. (b) sakldjfhsdfjh. 

### Conclusion/Discussion

The first theory of state-dependent perception bridging the scales from membrane potential dynamics to detection of sensory features.
A spike-based computation

References
[1] Arieli et al. (1996) Science, 273(5283), 1868-1871.
[2] McGinley et al. (2015) Neuron 
[] Jiang et al. (2015) Science, 350(6264), aac9462.
[] Neske et al. (2019) J. Neurosci.
[] Zerlaut et al. (2019) Cell Reports 
[] Cardin (2019) Cur. Op. Neurobiol.
[] The integrality of the simulations and analysis of the main and supplementary material are publicly available at the following address: https://github.com/yzerlaut/spkNtwk-cortStatesWakefulness/ 

### Results


[[Figure {Fig3} around here]]

### Section talking abour first result

We replace values coming from the "analysis.npz" file:

This statistical test was: {stat_test_example}
The output here was {data_output}

Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Etiam lobortis facilisis sem. Nullam necmi et neque pharetra sollicitudin. Praesent imperdiet mi nec ante. Donec ullamcorper, felis non sodales commodo, lectus velit ultrices augue, a dignissim nibh lectus placerat pede. Vivamus nuncnunc, molestie ut, ultricies vel, semper in, velit. Ut porttitor. Praesent in sapien. Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Duis fringilla tristique neque. Sed interdum libero ut metus. Pellentesque placerat. Nam rutrum augue a leo. Morbi sed elit sit amet ante lobortis sollicitudin. Praesent blandit blandit mauris. Praesent lectus tellus, aliquet aliquam, luctus a, egestas a, turpis. Mauris lacinia lorem sit amet ipsum. Nunc quis urna dictum turpis accumsan semper. \TODO{this additional analysis}

\begin{equation}
\label{eq:eq1}
\left\{
\begin{split}
& \frac{\partial^2 d}{\partial t ^2} = -x^3 \\
& \sum_{x} 1/x^2 \rightarrow y
\end{split}
\right.
\end{equation}


### Section talking abour second result

Lorem ipsum dolor sit amet, consectetuer adipisc- ing elit. Etiam lobortis facilisis sem. Nullam nec mi et neque pharetra sollicitudin. Praesent im- perdiet mi nec ante. Donec ullamcorper, felis non sodales commodo, lectus velit ultrices augue, a dig- nissim nibh lectus placerat pede. Vivamus nunc nunc, molestie ut, ultricies vel, semper in, velit.  Ut porttitor. Praesent in sapien. Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Duis fringilla tristique neque. Sed interdum libero ut metus. Pellentesque placerat. Nam rutrum augue a leo. Morbi sed elit sit amet ante lobortis sollicitudin. Praesent blandit blandit mauris. Praesent lectus tellus, aliquet aliquam, luctus a, egestas a, turpis. Mauris lacinia lorem sit amet ipsum. Nunc quis urna dictum turpis accumsan semper.Lorem ipsum dolor sit amet, consectetuer adipisc- ing elit. Etiam lobortis facilisis sem. Nullam nec mi et neque pharetra sollicitudin. Praesent im- perdiet mi nec ante. Donec ullamcorper, felis non sodales commodo, lectus velit ultrices augue, a dig- nissim nibh lectus placerat pede. Vivamus nunc nunc, molestie ut, ultricies vel, semper in, velit. Ut porttitor. Praesent in sapien. Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Duis fringilla tristique neque. Sed interdum libero ut metus. Pellentesque placerat. Nam rutrum augue a leo. Morbi sed elit sit amet ante lobortis sollici- tudin. Praesent blandit blandit mauris. Praesent lectus tellus, aliquet aliquam, luctus a, egestas a, turpis. Mauris lacinia lorem sit amet ipsum. Nunc quis urna dictum turpis accumsan semper.

* Methods

### First methodological section

\begin{equation}
\label{eq:first}
\tau \, \frac{dx}{dt} = E_L-v
\end{equation}

[[Table {Tab1} around here]]

[[Table {Tab2} around here]]

Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Etiam
lobortis facilisis sem. Nullam necmi et neque pharetra
sollicitudin. Praesent im-perdiet mi nec ante. Donec ullamcorper,
felis nonsodales commodo, lectus velit ultrices augue, a dig-nissim
nibh lectus placerat pede. Vivamus nuncnunc, molestie ut, ultricies
vel, semper in, velit.Ut porttitor. Praesent in sapien. Lorem
ipsumdolor sit amet, consectetuer adipiscing elit. Duisfringilla
tristique neque. Sed interdum libero utmetus. Pellentesque
placerat. Nam rutrum auguea leo. Morbi sed elit sit amet ante lobortis
sollici-tudin. Praesent blandit blandit mauris. Praesentlectus tellus,
aliquet aliquam, luctus a, egestas a,turpis. Mauris lacinia lorem sit
amet ipsum. Nuncquis urna dictum turpis accumsan semper.

### Second methodological section

\begin{equation}
\label{eq:second}
\tau \, \frac{dx}{dt} = E_L-v + \xi (t)
\end{equation}

Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Etiam
lobortis facilisis sem. Nullam necmi et neque pharetra
sollicitudin. Praesent im-perdiet mi nec ante. Donec ullamcorper,
felis nonsodales commodo, lectus velit ultrices augue, a dig-nissim
nibh lectus placerat pede. Vivamus nuncnunc, molestie ut, ultricies
vel, semper in, velit.Ut porttitor. Praesent in sapien. Lorem
ipsumdolor sit amet, consectetuer adipiscing elit. Duisfringilla
tristique neque. Sed interdum libero utmetus. Pellentesque
placerat. Nam rutrum auguea leo. Morbi sed elit sit amet ante lobortis
sollici-tudin. Praesent blandit blandit mauris. Praesentlectus tellus,
aliquet aliquam, luctus a, egestas a,turpis. Mauris lacinia lorem sit
amet ipsum. Nuncquis urna dictum turpis accumsan semper.  Lorem ipsum
dolor sit amet, consectetuer adipiscing elit. Etiam lobortis facilisis
sem. Nullam necmi et neque pharetra sollicitudin. Praesent im-perdiet
mi nec ante. Donec ullamcorper, felis nonsodales commodo, lectus velit
ultrices augue, a dig-nissim nibh lectus placerat pede. Vivamus
nuncnunc, molestie ut, ultricies vel, semper in, velit.Ut
porttitor. Praesent in sapien. Lorem ipsumdolor sit amet, consectetuer
adipiscing elit. Duisfringilla tristique neque. Sed interdum libero
utmetus. Pellentesque placerat. Nam rutrum auguea leo. Morbi sed elit
sit amet ante lobortis sollici-tudin. Praesent blandit blandit
mauris. Praesentlectus tellus, aliquet aliquam, luctus a, egestas
a,turpis. Mauris lacinia lorem sit amet ipsum. Nuncquis urna dictum
turpis accumsan semper.


playing with some reference here (Destexhe et al., 2003)

* Figures

### A spiking network model for the cortical states of wakefulness: architecture and dynamical regimes. 
#+options : {'label':'Fig1', 'extent':'singlecolumn', 'wrapfig':True, 'width':.57, 'height':10, 'wrapfig_space_left':-1.5, 'file':os.path.expanduser("~")+'/work/graphs/output/2d.png'}
(a) Schematic of the 5 population model.
(b) The U-model of cortical states.
(c) Minimization procedure.
(d) Resulting connectivity matrix.
(e) Network dynamics.

### Testing network function in two vision-inspired tasks.
#+options : {'label':'Fig3', 'extent':'doublecolumn', 'wrapfig':True, 'width':.52, 'height':9, 'hrule_bottom':True, 'file':os.path.expanduser("~")+'/work/graphs/output/fig.png'}
(a) A phenomenological model of the visual system up to V1 now feeds the local cortical model based on a screen content. This model  
(b) Praesentlectus tellus, aliquet aliquam, luctus a, egestas a,turpis. Mauris lacinia lorem sit amet ipsum. 
(c) Nuncquis urna dictum turpis accumsan semper. Lorem ipsum dolor sit amet, consectetuer adipisc ing elit. Etiam lobortis facilisis sem. Nullam nec mi et neque pharetra sollicitudin. Praesent imperdiet mi nec ante. Donec ullamcorper,felis non sodales commodo, lectus velit ultrices augue, a dignissim nibh lectus placerat pede. Vivamus nunc nunc, molestie ut, ultricies vel, semper in, velit.

*** Functional flexibility across network regimes: sparse activity regimes are optimal for natural scene identification while dense activity regimes are optimal for visual event detection.
#+options : {'label':'Fig3', 'extent':'doublecolumn', 'wrapfig':True, 'width':.52, 'height':9, 'hrule_bottom':True, 'file':os.path.expanduser("~")+'/work/graphs/output/fig.png'}
(a) Lorem ipsumdolor sit amet in *B* and *C*, consectetuer adipiscing elit. Duisfringilla tristique neque. Sed interdum libero utmetus. Pellentesque placerat. Nam rutrum auguea leo. Morbi sed elit sit amet ante lobortis sollici-tudin. Praesent blandit blandit mauris.
(b) Praesentlectus tellus, aliquet aliquam, luctus a, egestas a,turpis. Mauris lacinia lorem sit amet ipsum. 
(c) Nuncquis urna dictum turpis accumsan semper. Lorem ipsum dolor sit amet, consectetuer adipisc ing elit. Etiam lobortis facilisis sem. Nullam nec mi et neque pharetra sollicitudin. Praesent imperdiet mi nec ante. Donec ullamcorper,felis non sodales commodo, lectus velit ultrices augue, a dignissim nibh lectus placerat pede. Vivamus nunc nunc, molestie ut, ultricies vel, semper in, velit.

*** Role of rhythmic activity.
#+options : {'label':'Fig3', 'extent':'doublecolumn', 'wrapfig':True, 'width':.52, 'height':9, 'hrule_bottom':True, 'file':os.path.expanduser("~")+'/work/graphs/output/fig.png'}
(a) blabla.
(b) blabla.

* Tables

*** Model parameters
#+options : {'label':'Tab1', 'extent':'singlecolumn'}
Subcaption for the first table
| Name    | Description | Value | Unit |
|---------+-------------+-------+------|
|         | \textbf{}   |       |      |
|---------+-------------+-------+------|
| Peter   | {blabla}    |    17 |      |
| Anna    | 4321        |    25 |      |
| Patrick | 4321        |    25 |      |
|---------+-------------+-------+------|
\begin{tabular}{l|r|r}
Name & Phone & Age\\
\hline
Peter & {blabla} & 17\\
\hline
Anna & 4321 & 25\\
Patrick & 4321 & 25\\
\hline
\end{tabular}

*** Caption for the second table
#+options : {'label':'Tab2', 'extent':'doublecolumn'}
Subcaption for the second table
| model   | \(P_0\)(mV) | \(P_\mu\)(mV)             | \(P_\sigma\)(mV) | \(P_\tau\)(mV) |
|---------+-------------+---------------------------+------------------+----------------|
| simple  | 8           | 9                         |                4 |            387 |
|---------+-------------+---------------------------+------------------+----------------|
| complex | \(\pi/d^4\) | \(\frac{\pi}{\sqrt{28}}\) |               23 |              3 |
|---------+-------------+---------------------------+------------------+----------------|
| none    | 0           | 0                         |                0 |              0 |
|---------+-------------+---------------------------+------------------+----------------|
\begin{center}
\begin{tabular}{lrrrr}
model & \(P_0\)(mV) & \(P_\mu\)(mV) & \(P_\sigma\)(mV) & \(P_\tau\)(mV)\\
\hline
simple & 8 & 9 & 4 & 387\\
\hline
complex & \(\pi/d^4\) & \(\frac{\pi}{\sqrt{28}}\) & 23 & 3\\
\hline
none & 0 & 0 & 0 & 0\\
\hline
\end{tabular}
\end{center}

* Supplementary

** Supplementary Figures

*** Transfer function determination. 
#+options : {'label':'tf', 'extent':'singlecolumn', 'file':'figures/tf.png'}
(a) Example simulation of single cell activity with the simulated presynaptic spikes and resulting synaptic currents and $V_m$ fluctuations.
(b) Resulting transfer function and its analytical fit.

* Informations

*** Authors
Yann Zerlaut{1,2}
*** Short_Title
A spiking network model for the modulation of sensation
*** Short_Authors
Y. Zerlaut
*** Affiliations
{1} ICM, Brain and Spine Institute, Hôpital de la Pitié-Salpêtrière, Sorbonne Université, Inserm, CNRS, Paris, France. {2} Neuroinformatics group, Centre National de la Recherche Scientifique, Gif sur Yvette France.
*** Correspondence
yann.zerlaut@cnrs.fr
*** Keywords
spiking network model, cortical states, wakefulness, neural coding
*** Acknowledgements
I thank Andrew Davison, Nelson Rebola and Alberto Bacci for support and discussions. I thank [...] for feedback on the manuscript.
*** Funding
This research was supported by a postdoctoral fellowship from the Fondation pour la Recherche Médicale (No ARF201909009117). This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 892175 (InProSMod).
*** Data_Availability 
The integrality of the simulations and analysis of the main and supplementary material are publicly available at the following address: https://github.com/yzerlaut/spkNtwk-cortStatesWakefulness/
*** Conflict_Of_Interest
The author declares no conflict of interest.

* References

@Article{Cardin_2019,
  author          = {Cardin, Jessica A.},
  journal         = {Current opinion in neurobiology},
  title           = {Functional flexibility in cortical circuits.},
  year            = {2019},
  issn            = {1873-6882},
  month           = oct,
  pages           = {175--180},
  volume          = {58},
  citation-subset = {IM},
  completed       = {2020-02-11},
  country         = {England},
  doi             = {10.1016/j.conb.2019.09.008},
  issn-linking    = {0959-4388},
  keywords        = {Interneurons},
  mid             = {NIHMS1543446},
  nlm-id          = {9111376},
  owner           = {NLM},
  pii             = {S0959-4388(19)30040-6},
  pmc             = {PMC6981226},
  pmid            = {31585330},
  pubmodel        = {Print-Electronic},
  pubstate        = {ppublish},
  revised         = {2020-02-11},
}


@Article{McGinley_et_al_2015a,
  author    = {McGinley, Matthew J and David, Stephen V and McCormick, David A},
  title     = {Cortical membrane potential signature of optimal states for sensory signal detection},
  journal   = {Neuron},
  year      = {2015},
  volume    = {87},
  number    = {1},
  pages     = {179--192},
  doi       = {10.1016/j.neuron.2015.05.038},
  file      = {:mcginley2015cortical.pdf:PDF},
  publisher = {Elsevier},
}

@Article{McGinley_et_al_2015b,
  author    = {McGinley, Matthew J and Vinck, Martin and Reimer, Jacob and Batista-Brito, Renata and Zagha, Edward and Cadwell, Cathryn R and Tolias, Andreas S and Cardin, Jessica A and McCormick, David A},
  title     = {Waking state: rapid variations modulate neural and behavioral responses},
  journal   = {Neuron},
  year      = {2015},
  volume    = {87},
  number    = {6},
  pages     = {1143--1161},
  doi       = {10.1016/j.neuron.2015.09.012},
  file      = {:mcginley2015waking.pdf:PDF},
  publisher = {Elsevier},
}

* Other

** Pieces of abstract

Despite the large body of experimental work highlighting the critical
role of state-dependent computations in the awake cortex, a
theoretical framework providing a mechanistic and functional
description of this phenomenon is still lacking. The present study
builds a minimal spiking network model of a local cortical assembly
and examines its dynamical and computational properties. It is first
shown that a weak intrinsic rhythmicity coupled with competing
inhibitory and disinhibitory circuits allow recurrent spiking networks
to generate the set of dynamical regimes known as the “U-model” of
cortical states: the regimes observed at various arousal levels in
rodent sensory cortices. We then test the functional specificity of
such network states in two vision-inspired tasks. In a natural scene
discrimination task, the network encoding is found to be optimal
(high-reliability) when displaying the sparse activity regime
associated to intermediate arousal levels. In a low contrast detection
task, the network is optimally detecting (highly-sensitive) when
displaying the high-sensitivity dense activity regime found at high
arousal levels. Finally, the model proposes a functional role for the
rhythmic activity characterizing the idling cortex, it insures
unambiguous transitions between the high-reliability and
high-sensitivity modes of cortical activity. This theoretical analysis
suggests that the key functional constraint governing cortical
dynamics and function is the switch between high-reliablity and
high-sensitivity modes of recurrent dynamics and demonstrates that
spike-based interactions in recurrent networks allow to implement a
switch between high-reliability and high-sensitivity modes of
recurrent and suggests that such strategy.


This theoretical work demonstrates that spike-based interactions in
recurrent networks allow to implement a reliable switch between
high-reliability and high-sensitivity modes of signal processing. This
suggests that this principle is the key functional constraint
governing cortical operation in the awake cortex.

** Pieces of text
** Titles

- Adaptable/Shifting spiking dynamics enables flexible computation across waking states in sensory cortex
- Spiking network dynamics enables flexible computation across waking states in sensory cortex
- A Spiking Network Model for the Modulation of Cortical Dynamics and Function during Wakefulness in Sensory Cortex
