import os
import random
import string
import argparse
import subprocess
import itertools
import re

import pykrylov


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_PORT = 2020
DEFAULT_GRID_LR = ["2e-4", "1e-4", "5e-5"]
DEFAULT_GRID_BS = ["2", "4"]
DEFAULT_GRID_EPOCH = ["5", "10"]


def parse_args():
    parser = argparse.ArgumentParser(description="CLI Configuration")
    parser.add_argument("script", type=str, help="Which script to run")
    parser.add_argument("--ems_project", type=str, default="mnist-baliao")
    parser.add_argument("--exp_name", type=str, default="train")
    parser.add_argument("--cluster", type=str, default="tess137")
    parser.add_argument("--namespace", type=str, default="chatgpt-training-slc-a100")
    parser.add_argument("--image", type=str, default="hub.tess.io/baliao/dpc:base")
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--memory", type=int, default=64)
    parser.add_argument("--gpu_per_node", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gpu_model", type=str, default="a100")
    parser.add_argument(
        "--rack_name", default=None, type=str, help="Specify which rack to use"
    )
    parser.add_argument("--job_name", default="job-baliao-py-ba", type=str)
    parser.add_argument(
        "--grid-lr",
        nargs="+",
        default=DEFAULT_GRID_LR,
        help="Learning-rate grid values. Supports comma-separated and/or space-separated values.",
    )
    parser.add_argument(
        "--grid-bs",
        nargs="+",
        default=DEFAULT_GRID_BS,
        help="Batch-size grid values. Supports comma-separated and/or space-separated values.",
    )
    parser.add_argument(
        "--grid-epoch",
        nargs="+",
        default=DEFAULT_GRID_EPOCH,
        help="Epoch grid values. Supports comma-separated and/or space-separated values.",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable grid mode and submit a single job with --single-* values.",
    )
    parser.add_argument("--single-lr", type=str, default="2e-4")
    parser.add_argument("--single-bs", type=str, default="4")
    parser.add_argument("--single-epoch", type=str, default="10")
    return parser.parse_args()


def init_krylov_common_context():
    krylov_data_dir = "/mnt/mtrepo/data/baliao/hf_home"
    #(
    #    f"/data/{os.environ['KRYLOV_NAMESPACE']}/data/{os.environ['KRYLOV_PRINCIPAL']}"
    #)
    os.environ["KRYLOV_DATA_DIR"] = krylov_data_dir
    os.environ["HF_HOME"] = os.path.join(krylov_data_dir, ".lmm_cache/hf_cache")
    os.environ["HF_MODULES_CACHE"] = os.path.join(
        krylov_data_dir, ".lmm_cache/hf_module_cache"
    )
    os.environ["HF_DATASETS_CACHE"] = os.path.join(
        krylov_data_dir, ".lmm_cache/tmp/hf_data_cache"
    )
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(
        krylov_data_dir, ".lmm_cache/hf_cache"
    )
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(
        krylov_data_dir, ".lmm_cache/torch_cache"
    )
    # os.environ["TRITON_CACHE_DIR"] = os.path.join(krylov_data_dir, ".lmm_cache/.triton")
    # os.environ["PVC_MOUNT_PATH"] = os.path.join("/mnt", PVC_MOUNT_NAME)


def init_krylov_context():
    init_krylov_common_context()
    if "KRYLOV_WS_NAME" not in os.environ:
        # Get params
        context = pykrylov.util.get_task_context()
        if "experiment_id" in context:
            experiment_id = context["experiment_id"]
            # These 2 lines make task logs viewable via experiment view on aihub
            pykrylov.util.set_global_context(
                {pykrylov.util.consts.EXP_ID: experiment_id}
            )
            pykrylov.ems.experiment.update_experiment(
                experiment_id,
                runtime={"workflow": {"runId": os.environ["KRYLOV_WF_RUN_ID"]}},
            )
        return {
            'gpu_per_node': int(context['gpu_per_node']),
            'num_nodes': int(context['num_nodes']),
            'script': context['script'],
            'train_lr': context.get('train_lr'),
            'train_bs': context.get('train_bs'),
            'train_epoch': context.get('train_epoch'),
        }


def train():
    context = init_krylov_context()

    script_path = os.path.join(ROOT_DIR, context['script'])
    os.chmod(script_path, 755)

    train_bs = context.get("train_bs")
    train_lr = context.get("train_lr")
    train_epoch = context.get("train_epoch")

    run_env = os.environ.copy()
    if train_bs not in (None, ""):
        run_env["BS"] = str(train_bs)
    if train_lr not in (None, ""):
        run_env["LR"] = str(train_lr)
    if train_epoch not in (None, ""):
        run_env["EP"] = str(train_epoch)

    output = subprocess.run([script_path], check=True, env=run_env)

    if output.returncode != 0:
        raise ValueError(f"Script exited with error {output.returncode}")


def _split_csv_list(values):
    out = []
    for v in values:
        for x in str(v).split(","):
            x = x.strip()
            if x:
                out.append(x)
    return out


def _safe_tag(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _iter_grid(args):
    if args.no_grid:
        return [(str(args.single_lr), str(args.single_bs), str(args.single_epoch))]
    lrs = _split_csv_list(args.grid_lr)
    bss = _split_csv_list(args.grid_bs)
    eps = _split_csv_list(args.grid_epoch)
    if not lrs or not bss or not eps:
        raise ValueError("Grid values cannot be empty.")
    return list(itertools.product(lrs, bss, eps))


def _submit_one(args, script_basename, lr, bs, ep):
    # Sanity check
    if args.rack_name is not None:
        assert args.rack_name in ["slc_slc03_01-0200_11_20", "slc_slc03_01-0200_12_20"]

    tag = f"lr{_safe_tag(lr)}-bs{_safe_tag(bs)}-ep{_safe_tag(ep)}"
    experiment_id = None
    if not args.cluster == 'tess137':
        experiment_id = pykrylov.ems.experiment.create_experiment(
            args.ems_project, f"{args.exp_name}-{tag}"
        )

    master_name = "llm_" + "".join(random.choices(string.ascii_letters, k=8))
    master_service_name = master_name + "_svc"

    # Init pykrylov task
    if args.num_nodes > 1:
        task = DeepspeedTask(
            train,
            args=[],
            name=master_name,
            main_service_port=MASTER_PORT,
            gpu_per_worker=args.gpu_per_node,
            num_workers=args.num_nodes,
        )
        # task = pykrylov.distributed.DistributedTask(
        #     train,
        #     args=[],
        #     parallelism=args.num_nodes,
        #     name=master_name,
        #     service_name=master_service_name,
        #     service_port=MASTER_PORT,
        # )
        # task = pykrylov.contrib.tasks.torch.TorchTask(
        #     train,
        #     args=[],
        #     name=master_name,
        #     main_service_port=MASTER_PORT,
        #     gpu_per_worker=args.gpu_per_node,
        #     num_workers=args.num_nodes,
        # )
    else:
        task = pykrylov.Task(train, args=[])

    # Task setting
    task.add_task_parameters(
        {
            "ems_project": args.ems_project,
            "experiment_name": f"{args.exp_name}-{tag}",
            "gpu_per_node": args.gpu_per_node,
            "num_nodes": args.num_nodes,
            "master_name": master_name,
            "master_service_name": master_service_name,
            "master_port": MASTER_PORT,
            "experiment_id": experiment_id,
            "script": script_basename,
            "train_lr": str(lr),
            "train_bs": str(bs),
            "train_epoch": str(ep),
        }
    )
    task.set_image(args.image)
    task.run_on_gpu(args.gpu_per_node, model=args.gpu_model)
    task.add_cpu(args.cpu)
    task.add_memory(args.memory)
    task.add_file(args.script)
    task.add_execution_parameter("requireSameRack", "true")
    if args.cluster == "tess38":
        task.add_execution_parameter("nodeSelector", {"sku": "gpu3g10"})  # For H100

    if args.rack_name is not None:
        task.add_execution_parameter(
            "nodeSelector", {"failure-domain.tess.io/rack": args.rack_name}
        )

    if args.cluster == "tess40":
        task.mount_nfs("mtrepo", "10.5.1.56", "/krylov_shared_volume/krylov_shared")
    if args.cluster == "tess137":
        task.mount_pvc("mtrepo", "nlp-ebert-01", args.cluster)
        task.mount_pvc("nushare2", "krylov-user-pvc-nlp-01", args.cluster)
        #task.mount_pvc("nushare", "krylov-user-pvc-nlp-137", args.cluster)
    if args.cluster == "tess45":
        task.mount_pvc("nushare", "krylov-user-pvc-nlp-45", args.cluster)
        task.mount_pvc("mtrepo", "nlp-ebert-02", args.cluster)
        task.mount_pvc("nushare2", "krylov-user-pvc-nlp-01", args.cluster)
    if args.cluster == "tess38":
        task.mount_pvc("nushare2", "krylov-user-pvc-nlp-01", args.cluster)
        #task.mount_pvc("nushare", "krylov-user-pvc-nlp-38", args.cluster)
        task.mount_pvc("mtrepo", "nlp-ebert-02", args.cluster)

    # Submit workflow
    workflow = pykrylov.Flow(task)
    workflow.execution_parameters.add_execution_parameter("enableChooseCluster", "true")

    session = pykrylov.Session(namespace=args.namespace, job_name=f"{args.job_name}-{tag}")
    submitted_id = session.submit_experiment(
        workflow,
        project=args.ems_project,
        experiment_name=f"{args.exp_name}-{tag}",
        labels=[],
    )

    link = f"https://aip.vip.ebay.com/data/experiment-detail?projectName={args.ems_project}&experimentId={submitted_id}"
    print(f"[SUBMITTED] {tag} -> {link}")
    return link


def main(args):
    # Log in once
    user_name = os.popen("whoami").read().rstrip()
    pykrylov.util.config.use_account(user_name, yubikey_required=True)

    script_basename = os.path.basename(args.script)
    grid = _iter_grid(args)
    print(f"Submitting {len(grid)} job(s) from grid search...")
    for lr, bs, ep in grid:
        _submit_one(args, script_basename, lr, bs, ep)

    # Keep one canonical monitor line for compatibility
    link = "https://aip.vip.ebay.com/data/experiment-list"
    print(f"You can monitor progress and download result by visiting {link}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
