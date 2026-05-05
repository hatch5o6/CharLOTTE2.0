import submitit
import os
import json
from sloth_hatch.sloth import log_function_call

@log_function_call
def submit_slurm(
    function,
    job_name,
    output_folder,
    mail_user,
    timeout=24,
    ntasks_per_node=1,
    nodes=1,
    mem_gb=1,
    n_gpus=1,
    gpu_type="a100",
    mail_type="BEGIN,END,FAIL",
    qos="matrix",
    afterok=None # job id that must finish before this job starts
):
    os.makedirs(output_folder, exist_ok=True)
    fs = os.listdir(output_folder)
    for f in fs:
        if not f.isdecimal():
            raise ValueError(f"Names of directories in `{output_folder}` must be integers!")
    if len(fs) > 0:
        next_dir = str(max([int(f) for f in fs]) + 1)
    else:
        next_dir = "0"
    next_dir = os.path.join(output_folder, next_dir)
    assert not os.path.exists(next_dir)

    executor = submitit.AutoExecutor(folder=next_dir)

    if n_gpus > 0:
        gpu_type = gpu_type.strip()
        gpus = f"{gpu_type}:{n_gpus}" if gpu_type else f"{n_gpus}"
    else:
        if n_gpus != 0:
            raise ValueError("n_gpus must be >= 0!")
        gpus = "0"

    additional_params = {
        "gpus": f"{gpus}",
        "ntasks-per-node": ntasks_per_node,
        "qos": qos,
        "mail-type": mail_type,
        "mail-user": mail_user
    }
    if afterok is not None:
        additional_params["dependency"] = f"afterok:{afterok}"
    print("################## SLURM ##################")
    print(f"\tTIMEOUT: {timeout}")
    print(f"\tNODES: {nodes}")
    print(f"\tMEM_GB: {mem_gb}")
    print(f"\tJOB_NAME: {job_name}")
    print("Additional Slurm Parameters:\n", json.dumps(additional_params, indent=2))
    executor.update_parameters(
        timeout_min=timeout,
        nodes=nodes,
        mem_gb=mem_gb,
        slurm_job_name=job_name,
        slurm_additional_parameters=additional_params
    )
    job = executor.submit(function)
    print(f"Submitted {job_name} job ({job.job_id})")
    print("###########################################")
    return job
