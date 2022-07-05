## Distributed Deep RL training using RLLIB on Azure. 

The code in this repository acts as a sample project for running distributed reinforcement learning jobs on Azure using Ray's RLLIB.
To run the project from your machine, the following steps should be satisfied. 

### Pre-requisities
- Azure ML Workspace 
- Compute instance for triggering Job
- Compute cluster for running the training job.


### Steps
- Login to your Azure account and create an Azure ML workspace. 
- Create a compute instance (LINUX)
- git clone this repo into the workspace.
- Run the below command to create a compute cluster
```
python infra/cluster.py
```
- To run the single agent DQN training, run the below command
```
python dqn.py
``` 
- To run the training Job, run the below command
```
python run_experiment.py
```

Optional
- You can run the RL job in local machine by running the below command. In local mode the script uses the developer workstation to spin-off workers (1 by default).
```
python run_tune_local.py
```
