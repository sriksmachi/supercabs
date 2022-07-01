import json
from ray_on_aml.core import Ray_On_AML
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core import Workspace, Environment, ScriptRunConfig, Experiment
from azureml.core.runconfig import EnvironmentDefinition
from azureml.widgets import RunDetails
from azureml.tensorboard import Tensorboard
from ray_on_aml.core import Ray_On_AML
from azureml.widgets import RunDetails
from azureml.core.runconfig import RunConfiguration, DockerConfiguration


def main():
    ws = Workspace.from_config()
    experiment_name = "contoso_cabs_on_ray"
    compute_name = 'ray-cpu-cluster'

    # Run infra/cluster.py for creating the cluster
    compute_target = ws.compute_targets[compute_name]
    rayEnv = Environment.from_conda_specification(name = "RLEnv",
                                             file_path = "ray_job_env.yml")
    rayEnv.docker.base_image = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04:20220329.v1"
    experiment = Experiment(ws, experiment_name)
    aml_run_config_ml = RunConfiguration(communicator='OpenMpi')
    aml_run_config_ml.target = compute_target
    aml_run_config_ml.docker = DockerConfiguration(use_docker=False)
    aml_run_config_ml.node_count = 3
    aml_run_config_ml.environment = rayEnv

    training_algorithm = "APEX"
    rl_environment = "contosocabs-v0"
    script_name='run_tune_job.py'

    config = {
        "num_gpus": 0,
        "num_workers": 2,
        "gamma": 0.98,
        "n_step": 10,
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 1,
    } 

    command=[
        'python', script_name,
        '--run', training_algorithm,
        '--env', rl_environment,
        '--config', json.dumps(json.dumps(config)),
        '--stop', '\'{"episode_reward_mean": 1000, "time_total_s": 3600}\''
    ]

    src = ScriptRunConfig(source_directory ='.',
                        command = command,
                        run_config = aml_run_config_ml
                       )

    run = experiment.submit(src)

if __name__ == "__main__":
    main()