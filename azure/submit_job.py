from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, AmlCompute

def submit_training_job():
    """
    Submits the PyTorch SFT training job to an Azure ML GPU cluster.
    """
    print("🚀 Authenticating with Azure...")
    credential = DefaultAzureCredential()
    
    # Initialize MLClient (Replace with actual subscription ID, resource group, workspace)
    try:
        ml_client = MLClient(
            credential=credential,
            subscription_id="YOUR_SUBSCRIPTION_ID",
            resource_group_name="YOUR_RESOURCE_GROUP",
            workspace_name="YOUR_WORKSPACE_NAME"
        )
    except Exception as e:
        print("⚠️ Authentication failed (expected if not running in an authenticated Azure context).")
        print(f"Details: {e}")
        return

    print("☁️ Defining Environment...")
    env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
        conda_file="environment.yml",
        name="finance-slm-env"
    )

    print("🖥️ Defining Compute Target (e.g., NC6s_v3 for GPU)...")
    # Assuming the compute cluster 'gpu-cluster' is already created in the workspace.
    compute_target = "gpu-cluster" 

    print("📦 Defining Command Job...")
    job = command(
        code="../training",           # Directory containing the training script
        command="python train.py",    # Command to run
        environment=env,
        compute=compute_target,
        display_name="finance_slm_sft_training",
        experiment_name="SLM_FineTuning"
    )

    print("📤 Submitting Job to Azure ML...")
    returned_job = ml_client.create_or_update(job)
    print(f"✅ Job submitted successfully! Studio URL: {returned_job.studio_url}")

if __name__ == "__main__":
    submit_training_job()
