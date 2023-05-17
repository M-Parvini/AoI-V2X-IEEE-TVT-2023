
# Simulation code of the paper:
    "AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via Multi-Agent Multi-Task Reinforcement Learning"

### If you want to cite: 
>M. Parvini, M. R. Javan, N. Mokari, B. Abbasi and E. A. Jorswieck, "AoI-Aware Resource Allocation for Platoon-Based C-V2X Networks via Multi-Agent Multi-Task Reinforcement Learning," in IEEE Transactions on Vehicular Technology, doi: 10.1109/TVT.2023.3259688.

# Simulation environment is based on the urban case defined in Annex A of 
     3GPP, TS 36.885, "Study on LTE-based V2X Services".

# prerequisites:
python 3.7 or higher
PyTorch 1.7 or higher + CUDA
It is recommended that the latest drivers be installed for the GPU.
---------------------------------------------------------------------------------------

In order to run the code:
***
Please make sure that you have created the following directories:
	1) ...\Classes\tmp\ddpg
	2) ...\model\marl_model

The final results and the network weights will be saved in these directories.
***

1- Change the number of vehicles, platoon sizes, and intra-platoon distance

2- Once you run the code, simulation results will be saved into the directory: 
   ...\model\marl_model. You can import these data wherever you want (Matlab, python, etc.) 
   and plot the results. Furthermore, the weights of the neural networks will be saved into 
   the directory: ...\Classes\tmp\ddpg. 

3- Except for Fig. 1, which can be directly obtained through reward_t1.mat and reward_t2.mat, 
   in order to plot the other figures, the results should be averaged with respect to the agents.

4- Figs. 2 and 3, are plotted as follows:

	Run Modified MADDPG with TDec\Main, and average the results of (reward_t1.mat+reward_t2.mat) 
	for all the agents to produce the first plot. 
	
	Run Modified MADDPG\Main, and average the results of reward.mat for all the agents to produce 
	the second plot.
	
	Run MADDPG_FDec\Main, and average the results of reward.mat for all the agents to produce the 
	third plot.
	
	Run DDPG\Main, and average the results of reward.mat for all the agents to produce the forth plot.

5- The remaning figures can be reproduced by the same procedure.
