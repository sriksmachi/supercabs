
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core import Workspace, Environment, ScriptRunConfig, Experiment
from azureml.core.runconfig import EnvironmentDefinition
from azureml.widgets import RunDetails
from azureml.tensorboard import Tensorboard

from ray_on_aml.core import Ray_On_AML
from azureml.widgets import RunDetails
from azureml.core.runconfig import RunConfiguration

# choose a name for the Ray head cluster
vnet_name = 'rl_gym_vnet'

def create_head(ws):
    head_compute_name = 'head-gpu'
    head_compute_min_nodes = 0
    head_compute_max_nodes = 2
    # This example uses GPU VM. 
    # For using CPU VM, set SKU to STANDARD_D2_V2
    head_vm_size = 'STANDARD_NC6'
    if head_compute_name in ws.compute_targets:
        head_compute_target = ws.compute_targets[head_compute_name]
        if head_compute_target and type(head_compute_target) is AmlCompute:
            print(f'found head compute target. just use it {head_compute_name}')
    else:
        print('creating a new head compute target...')
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = head_vm_size,
                                                                    min_nodes = head_compute_min_nodes, 
                                                                    max_nodes = head_compute_max_nodes,
                                                                    vnet_resourcegroup_name = ws.resource_group,
                                                                    vnet_name = vnet_name,
                                                                    subnet_name = 'default')

        # create the cluster
        head_compute_target = ComputeTarget.create(ws, head_compute_name, provisioning_config)

        # can poll for a minimum number of nodes and for a specific timeout. 
        # if no min node count is provided it will use the scale settings for the cluster
        head_compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

         # For a more detailed view of current AmlCompute status, use get_status()
        print(head_compute_target.get_status().serialize())


def create_worker(ws):
    # choose a name for your Ray worker cluster
    worker_compute_name = 'worker-cpu'
    worker_compute_min_nodes = 0 
    worker_compute_max_nodes = 4

    # This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
    worker_vm_size = 'STANDARD_D2_V2'

    # Create the compute target if it hasn't been created already
    if worker_compute_name in ws.compute_targets:
        worker_compute_target = ws.compute_targets[worker_compute_name]
        if worker_compute_target and type(worker_compute_target) is AmlCompute:
            print(f'found worker compute target. just use it {worker_compute_name}')
    else:
        print('creating a new worker compute target...')
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = worker_vm_size,
                                                                    min_nodes = worker_compute_min_nodes, 
                                                                    max_nodes = worker_compute_max_nodes,
                                                                    vnet_resourcegroup_name = ws.resource_group,
                                                                    vnet_name = vnet_name,
                                                                    subnet_name = 'default')

        # create the cluster
        worker_compute_target = ComputeTarget.create(ws, worker_compute_name, provisioning_config)

        # can poll for a minimum number of nodes and for a specific timeout. 
        # if no min node count is provided it will use the scale settings for the cluster
        worker_compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

         # For a more detailed view of current AmlCompute status, use get_status()
        print(worker_compute_target.get_status().serialize())

def main():
    ws = Workspace.from_config()
    create_head(ws)
    create_worker(ws)

if __name__ == "__main__":
    main()