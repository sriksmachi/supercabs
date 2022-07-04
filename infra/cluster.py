
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core import Workspace, Environment, ScriptRunConfig, Experiment
from azureml.core.runconfig import EnvironmentDefinition
from azureml.widgets import RunDetails
from azureml.tensorboard import Tensorboard

from ray_on_aml.core import Ray_On_AML
from azureml.widgets import RunDetails
from azureml.core.runconfig import RunConfiguration


def create_cluster(ws):
    compute_name = 'ray-cluster-cpu'
    compute_min_nodes = 0
    compute_max_nodes = 2

    # This example uses GPU VM. For using CPU VM, set SKU to STANDARD_D2_V2
    vm_size = 'STANDARD_D2_V2'

    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            if compute_target.provisioning_state == 'Succeeded':
                print('found compute target. just use it', compute_name)
            else: 
                raise Exception(
                    'found compute target but it is in state', compute_target.provisioning_state)
    else:
        print('creating a new compute target...')
        provisioning_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size,
            min_nodes=compute_min_nodes, 
            max_nodes=compute_max_nodes,
        )

        # Create the cluster
        compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

        # Can poll for a minimum number of nodes and for a specific timeout. 
        # If no min node count is provided it will use the scale settings for the cluster
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

        # For a more detailed view of current AmlCompute status, use get_status()
        print(compute_target.get_status().serialize())

def main():
    ws = Workspace.from_config()
    create_cluster(ws)

if __name__ == "__main__":
    main()