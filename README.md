# Contraction Actor-Critic (Contraction Metric Guided Reinforcement Learning for Robust Path-Tracking)
![cac](https://github.com/user-attachments/assets/c273cd24-14d8-4172-97d4-67b4e2b3dd6a)

It integrates a CCM into reinforcement learning (RL), where the CCM serves as a critic in learning control policies that minimize cumulative tracking error under unknown dynamics. 
Given a pre-trained dynamics model, CAC simultaneously learns a contraction metric generator (CMG)—which finds a Riemannian metric under which a system contracts—and uses an actor-critic algorithm to learn an optimal tracking policy guided by that contracting geometry.
The code supports multiple benchmark environments—Car, PVTOL, NeuralLander, and Quadrotor—for extensive numerical simulations.

## Authors
* **Minjae Cho** - _The Grainger College of Engineering, University of Illinois Urbana-Champaign_
* **Hiroyasu Tsukamoto** - _The Grainger College of Engineering, University of Illinois Urbana-Champaign_
* **Huy T. Tran** - _The Grainger College of Engineering, University of Illinois Urbana-Champaign_
  
If ones have questions about this project, please reach out to Minjae.


## Prerequisites
In any local folder, open a terminal and run the following command to download our package into the current directory:
```
git clone https://github.com/Mgineer117/CAC/
cd CAC
```

We assume that you have Conda installed. If not, please refer to the [Anaconda installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install). Our code is compatible with Python version > 3.10. We recommend creating a dedicated virtual environment as follows:
```
conda create -n CAC python==3.12.*
conda activate CAC
```

Then, install the required Python packages using:
```
pip install -r requirements.txt
```

## Getting Started
We describe our code structure for those who want to develop it further for their research.
```
├── config **[Contain environment specific parameters for each algorithm]**
├── envs **[Contain simulator for each environments]**
├── log **[Contain logger for both tensorboard and wandb]**
├── model **[Contain Neurallander dynamic parameters and used to load other parameters too]**
├── policy **[Contains the implementation of algorithms]**
│   ├── layers **[Contrains a backbone of algorithms]**
│   ├── base.py
│   ├── c3m.py
│   ├── cac.py
│   ├── lqr.py
│   ├── ppo.py
│   └── sd_lqr.py
├── trainer **[Contain an online trainer for training akgorithms]**
├── utils **[Contain util functions and sampler to collect online data]**
├── README.md
├── requirements.txt
├── evaluate.py [Contain code to evaluate given *.pth model]
├── main.py [Main file to run experiment]
```

## Training
Our codebase uses the following command to regulate the running parameters,
```
python3 main.py --task car --algo-name cac-approx
```
or
```
python3 main.py --task neurallander --algo-name cac
```
where all arguments are written all lowercase and ```-approx``` behind algorithm name denotes whethere one wants to run it with approximated dynamics (for unknown dynamics assumptions).

## Evaluation 
After one trains the algorithm, the *.pth file for that algorithm will be generated in the logging folder. One can evaluate the model by moving the model to ```model/``` directory and running the ```evaluate.py```.


## Robot Demonstration Video
One can refer to the robot experiment result [here](https://youtu.be/jJnogKxIXfI).

## Logging
We support three logging options—Weights & Biases (WandB), TensorBoard, and CSV—to accommodate different user preferences. Specifically, when WandB is properly configured on your local machine, all algorithmic and parameter settings, along with real-time training metrics, are automatically logged to your WandB dashboard. Simultaneously, training results are saved locally in TensorBoard format for visualization, and evaluation metrics are exported as CSV files for easy analysis. In addition, the model parameters are saved along with the best-performing one.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
