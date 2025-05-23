# ANCM 2024 – final project

> Emergence of Linguistic Communication from Referential Games with Symbolic and Pixel Input

## Running with `uv`
```uv python install 3.8
uv venv --python 3.8
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .```

Running Jupyter Lab: `uv run --with jupyter jupyter lab`

## Training

```source .venv/bin/activate
python -m src.train --output_dir runs/ --filename run --data_path src/data/input_data/visa-5-256.npz ...```

## Figures

Script used to generate all figures is in the notebook `src/plots.ipynb`

```source .venv/bin/activate
uv run --with jupyter jupyter lab
```

## Contents
- `ancm/data/` data export module,
- `ancm/data/input_data` exported intput datasets
- `ancm/runs/` contains raw results
- `ancm/results/` contains plots and analyses

## Setup
```bash
python3 -m pip install -r ancm/requirements.txt
cd .. && python3 -m pip install --editable egg/ && cd ancm/
```

## Running locally

1. Set the desired hyperparameter values in `ancm/jobs/get_params.py`
2. Export the parameter file `python3 -m ancm.jobs.get_params params.txt`
3. Execute `python3 run_all.py results/ params.txt`
3. Edit the training script `ancm/jobs/job.sh`
4. Add the job `sbatch ancm/jobs/job.sh`

## Snellius

1. Set the desired hyperparameter values in `ancm/jobs/get_params.py`
2. Export the parameter file `python3 -m ancm.jobs.get_params ancm/jobs/params.txt`
3. Edit the training script `ancm/jobs/job.sh`
4. Add the job `sbatch ancm/jobs/job.sh`

## Plots & alalyses

`python3 -m ancm.analyse -i ancm/runs/preliminary_results -o ancm/results/preliminary_results [--recompute]`

## Training

Check out `run_test.sh`.

## Data

Exporting VISA to `ancm/data/input_data/visa-<n_distractors>-<n_samples_train>.npz`:
```bash
pytho3 -m ancm.data.visa -d <n_distractors> --n_samples_train <n_samples_train> 
```

A sample is a set of target concept + distractor concepts, sampled from the same category or the same categories (that can be changed by editing `reformat_visa.py`). 

| **Category** | **Num. concepts**  |
|--------------|-------------------:|
| animals      | 135                |
| appliances   | 18                 |
| artefacts    | 37                 |
| clothing     | 40                 |
| container    | 11                 |
| device       | 8                  |
| food         | 59                 |
| home         | 49                 |
| instruments  | 19                 |
| material     | 4                  |
| plants       | 7                  |
| structures   | 26                 |
| tools        | 27                 |
| toys         | 3                  |
| vehicles     | 37                 |
| weapons      | 23                 |





# EGG 🐣: Emergence of lanGuage in Games

![GitHub](https://img.shields.io/github/license/facebookresearch/EGG)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Introduction

EGG is a toolkit that allows researchers to quickly implement multi-agent games with discrete channel communication. In 
such games, the agents are trained to communicate with each other and jointly solve a task. Often, the way they communicate is not explicitly determined, allowing agents to come up with their own 'language' in order to solve the task.
Such setup opens a plethora of possibilities in studying emergent language and the impact of the nature of task being solved, the agents' models, etc. This subject is a vibrant area of research often considered as a prerequisite for general AI. The purpose of EGG is to offer researchers an easy and fast entry point into this research area.

EGG is based on [PyTorch](https://pytorch.org/) and provides: (a) simple, yet powerful components for implementing 
communication between agents, (b) a diverse set of pre-implemented games, (c) an interface to analyse the emergent 
communication protocols.

Key features:
 * Primitives for implementing a single-symbol or variable-length, discrete or continuous communication (with vanilla FFNs, RNNs, GRUs, LSTMs or Transformers);
 * Support for a single pair as well as a population of interacting agents;
 * Training with optimization of the communication channel with REINFORCE or Gumbel-Softmax relaxation via a common interface;
 * Simplified configuration of the general components, such as checkpointing, optimization, tensorboard support, etc;
 * Provides a simple CUDA-aware command-line tool for grid-search over parameters of the games.

To fully leverage EGG one would need at least [a high-level familiarity](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
with PyTorch. However, to get a taste of communication games without writing any code, you could try [a dataset-based game](/egg/zoo/external_game), which allow you to experiment with different signaling games by simply editing input files. 

## List of implemented games

<details><summary>List of example and tutorial games</summary><p>
 
 * [`MNIST autoencoder tutorial`](/tutorials/EGG%20walkthrough%20with%20a%20MNIST%20autoencoder.ipynb): A Jupyter tutorial that implements a MNIST discrete auto-encoder step-by-step, covering the basic concepts of EGG. The tutorial starts with pre-training a "vision" module and builds single- and multiple symbol auto-encoder communication games with channel optimization done by Reinforce and Gumbel-Softmax relaxation ([notebook](/tutorials/EGG%20walkthrough%20with%20a%20MNIST%20autoencoder.ipynb) / [colab](https://colab.research.google.com/github/facebookresearch/EGG/blob/main/tutorials/EGG%20walkthrough%20with%20a%20MNIST%20autoencoder.ipynb)).
 
* [`egg/zoo/basic_games`](/egg/zoo/basic_games): Simple implementations of reconstruction and discrimination games, taking their inputs from text files, and with thoroughly annotated code. These might be a good starting point to learn to play with EGG.
 
 * [`egg/zoo/signal_game`](/egg/zoo/signal_game): Modern version of a Lewis' signaling game. In this game, Sender is presented with a target image and with one or more distractor images. Then all images are shuffled and Receiver has to point to the target image based on a message from Sender. This implementation is based on Diane Bouchacourt's code.
 
 * [`egg/zoo/simple_autoenc`](/egg/zoo/simple_autoenc): Discrete auto-encoder Sender/Receiver game that auto-encodes one-hot vectors using variable-length messages.
 * [`egg/zoo/mnist_autoenc`](/egg/zoo/mnist_autoenc): Discrete MNIST auto-encoder game. In this Sender/Receiver game, Sender looks onto a MNIST image and sends a single symbol to Receiver, who tries to recover the image.
  * [`egg/zoo/mnist_vae`](/egg/zoo/mnist_vae): Continuous-message VAE cast as an auto-encoder game. In this Sender/Receiver (Encoder/Decoder) game, Sender looks onto a MNIST image and sends a multi-dimensional vector to Receiver, who tries to recover the image.
 * [`egg/zoo/summation`](/egg/zoo/summation): Sender and Receiver are jointly trained to recognize the `a^nb^n` grammar: Sender reads an input sequence and Receiver answers if the sequence belongs to the grammar. Which agent actually counts, Sender or Receiver? Does Sender make the decision and send it to Receiver? Or does Sender encode the incoming sequence in the message and it is Receiver that make the decision? Or something in-between?
 * [`egg/zoo/external_game`](/egg/zoo/external_game): A signaling game that takes inputs and ground-truth outputs from CSV files. 
</p></details>

 <details><summary>Games used in published work</summary><p>
 
  * [`egg/zoo/channel`](/egg/zoo/channel): _Anti-efficient encoding in emergent communication._ Rahma Chaabouni, Eugene Kharitonov, Emmanuel Dupoux, Marco Baroni. NeurIPS 2019.
  
  * [`egg/zoo/objects_game`](/egg/zoo/objects_game): _Focus on What’s Informative and Ignore What’s not: Communication Strategies in a Referential Game._ Roberto Dessì, Diane Bouchacourt, Davide Crepaldi, Marco Baroni. NeurIPS Workshop on Emergent Communication 2019. A Sender/Receiver game where the Sender sees a target as a vector of discrete properties (*e.g.* [2, 4, 3, 1] for a game with 4 dimensions) and the Receiver has to recognize the target among a lineup of target+distractor(s).
  
  * [`egg/zoo/compo_vs_generalization`](egg/zoo/compo_vs_generalization) _Compositionality and Generalization in Emergent Languages._ Rahma Chaabouni, Eugene Kharitonov, Diane Bouchacourt, Emmanuel Dupoux, Marco Baroni. ACL 2020.

  * [`egg/zoo/compo_vs_generalization_ood`](egg/zoo/compo_vs_generalization_ood) _Defending Compositionality in Emergent Languages._ Michal Auersperger, Pavel Pecina. NAACL SRW 2022.
  
  * [`egg/zoo/language_bottleneck`](/egg/zoo/language_bottleneck) _Entropy Minimization In Emergent Languages._ Eugene Kharitonov, Rahma Chaabouni, Diane Bouchacourt, Marco Baroni. ICML 2020. `egg/zoo/language_bottleneck` contains a set of games that study the information bottleneck property of the discrete communication channel. This poperty is illustrated in an EGG-based example of MNIST-based style transfer without an adversary ([notebook](/egg/zoo/language_bottleneck/mnist-style-transfer-via-bottleneck.ipynb) / [colab](https://colab.research.google.com/github/facebookresearch/EGG/blob/main/egg/zoo/language_bottleneck/mnist-style-transfer-via-bottleneck.ipynb)).

</p></details>

We are adding games all the time: please look at the [`egg/zoo`](/egg/zoo) directory to see what is available right now. Submit an issue if there is something you want to have implemented and included.

More details on each game's command line parameters are provided in the games' directories.

### An important technical point

EGG supports Reinforce and Gumbel-Softmax optimization of the *communication channel*. Currently, Gumbel-Softmax-based optimization is only supported if the game *loss* is differentiable. The [MNIST autoencoder game tutorial](/tutorials/EGG%20walkthrough%20with%20a%20MNIST%20autoencoder.ipynb) illustrates both Reinforce and Gumbel-Softmax channel optimization when using a differentiable game loss. The [signaling game](/egg/zoo/signal_game) has a non-differentiable game loss, and the communication channel is optimized with Reinforce.

## Installing EGG

Generally, we assume that you use PyTorch 1.1.0 or newer and Python 3.6 or newer. 

 1. (optional) It is a good idea to develop in a new conda environment, e.g. like this:
    ```
    conda create --name egg36 python=3.6
    conda activate egg36
    ```
 2. EGG can be installed as a package to be used as a library
    ```
    pip install git+ssh://git@github.com/facebookresearch/EGG.git
    ```
    or via https
    ```
    pip install git+https://github.com/facebookresearch/EGG.git
    ```
    Alternatively, EGG can be cloned and installed in editable mode, so that the copy can be changed:
    ```
    git clone git@github.com:facebookresearch/EGG.git && cd EGG
    pip install --editable .
    ```
 3.
    Then, we can run a game, e.g. the MNIST auto-encoding game:
    ```bash
    python -m egg.zoo.mnist_autoenc.train --vocab=10 --n_epochs=50
    ```

## EGG structure

The repo is organised as follows:
```
- tests # tests for the common components
- docs # documentation for EGG
- egg
-- core # common components: trainer, wrappers, games, utilities, ...
-- zoo  # pre-implemented games 
-- nest # a tool for hyperparameter grid search
```

## How-to-Start and Learning more
* Our EMNLP'19 Demo [paper](https://aclanthology.org/D19-3010/) provides a high-level view of the toolkit and points to further resources.
* The step-by-step [`MNIST autoencoder tutorial`](/tutorials/EGG%20walkthrough%20with%20a%20MNIST%20autoencoder.ipynb) goes over all essential steps to create a full-featured communication game with variable length messages between the agents. NB: depending on your computational resources, this might take a while to run! [(open in colab)](https://colab.research.google.com/github/facebookresearch/EGG/blob/main/tutorials/EGG%20walkthrough%20with%20a%20MNIST%20autoencoder.ipynb)
* The simplest starter code is in [`egg/zoo/basic_games`](/egg/zoo/basic_games), providing implementations of basic reconstruction and discrimination games. Input can be provided through text files, and the code is thoroughly commented.
* Another good starting point to implement a Sender/Receiver game is the MNIST autoencoder
game, [MNIST auto-encoder game](/egg/zoo/mnist_autoenc). The game features both Gumbel-Softmax 
and Reinforce-based implementations.
* A template for a game is provided in [`egg/zoo/template`](/egg/zoo/template).
* EGG can be used for autoencoder-based experiments or even self-supervised learning. An implementation of SimCLR introduce in Chen et al. 2020 can be found in [`egg/zoo/simclr`](/egg/zoo/simclr).
* EGG provides some utility boilerplate around commonly used command line parameters. Documentation about using it can be found
[here](docs/CL.md).
* A brief how-to for tensorboard is [here](docs/tensorboard.md).
* To learn more about the provided hyperparameter search tool, read this [doc](docs/nest.md).

## Citation
If you find EGG useful in your research, please cite this repository:
```
@misc{kharitonov:etal:2021,
  author = "Kharitonov, Eugene  and Dess{\`i}, Roberto and Chaabouni, Rahma  and Bouchacourt, Diane  and Baroni, Marco",
  title = "{EGG}: a toolkit for research on {E}mergence of lan{G}uage in {G}ames",
  howpublished = {\url{https://github.com/facebookresearch/EGG}},
  year = {2021}
}
```

## Contributing
Please read the contribution [guide](.github/CONTRIBUTING.md).


## Testing
Run pytest:

```
python -m pytest
```

All tests should pass.

## Licence
The majority of EGG is licensed under MIT, however portions of the project are available under separate license terms: [LARC](egg/zoo/emcom_as_ssl/LARC.py) is licensed under the BSD 3-Clause license.

The text of the license for EGG can be found [here](LICENSE).
