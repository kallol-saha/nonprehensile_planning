import wandb

# Helper function to find run ID by name
def find_run_id(project_name, run_name):
    api = wandb.Api()
    runs = api.runs(project_name)
    for run in runs:
        if run.name == run_name:
            return run.id
    return None

def start_wandb_run(args):
    run_name = args.wandb_run_name
    run_id = find_run_id(args.wandb_project_name, run_name)
    if run_id:
        print(f"Resuming run: {run_name} (ID: {run_id})")
        wandb.init(project=args.wandb_project_name, id=run_id, resume="must")
    else:
        print(f"Creating new run: {run_name}")
        wandb.init(project=args.wandb_project_name, name=run_name)
    # Save args to wandb
    wandb.config.update(args)