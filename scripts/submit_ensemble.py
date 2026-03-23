# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Submit ensemble.py as N parallel AML Command jobs, one per shard.

Workspace connection is read from environment variables (or .env):
    AML_SUBSCRIPTION_ID
    AML_RESOURCE_GROUP
    AML_WORKSPACE_NAME
    AML_COMPUTE_NAME     — default compute target
    AML_ENVIRONMENT_NAME — registered AML environment, e.g. "my-env:1.0"
    AZURE_TENANT_ID      — tenant ID for AzureCliCredential

SAS tokens (OUTPUT_SAS, UDM_SAS, BLOB_SAS) are read from the same .env and injected
into each job's environment_variables block.

Output datastore
----------------
Use --aml-output-path to write results directly to blob storage via a registered
AML datastore:

    --aml-output-path "azureml://datastores/<datastore>/paths/<prefix>/"

Files will land at:
    <prefix>/<last_timestamp>/<save_folder>/<scene>.tif

Omit --aml-output-path to write to the job's local ./outputs/ folder instead
(collected as job artifacts in AML Studio).

Example — full-scale parallel run (98 shards):

    python scripts/submit_ensemble.py \\
        --input-csv data/scenes.csv \\
        --model <model-name> \\
        --save-dir . \\
        --save-folder <run-name> \\
        --timestamps 2023q3 2023q4 2024q1 2024q2 \\
        --num-shards 98 \\
        --workers 32 \\
        --skip-existing \\
        --aml-output-path "azureml://datastores/<datastore>/paths/<prefix>/"

Single-shard test (1 job, all scenes):

    python scripts/submit_ensemble.py \\
        --input-csv data/scenes_test.csv \\
        --model <model-name> \\
        --save-dir . \\
        --save-folder <run-name> \\
        --timestamps 2023q3 2023q4 2024q1 2024q2 \\
        --num-shards 1 \\
        --workers 32 \\
        --aml-output-path "azureml://datastores/<datastore>/paths/<prefix>/"
"""

import argparse
import os
from pathlib import Path

from azure.ai.ml import MLClient, Output, command
from azure.ai.ml.entities import Environment
from azure.identity import AzureCliCredential, DefaultAzureCredential
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------

def _get_ml_client(args) -> MLClient:
    subscription_id = args.subscription_id or os.environ["AML_SUBSCRIPTION_ID"]
    resource_group  = args.resource_group  or os.environ["AML_RESOURCE_GROUP"]
    workspace_name  = args.workspace       or os.environ["AML_WORKSPACE_NAME"]
    tenant_id       = args.tenant_id       or os.environ.get("AZURE_TENANT_ID")
    credential = AzureCliCredential(tenant_id=tenant_id) if tenant_id else DefaultAzureCredential()
    return MLClient(credential, subscription_id, resource_group, workspace_name)


def _build_command(shard_index: int, args) -> str:
    """Build the ensemble.py CLI invocation for a single shard."""
    timestamps = " ".join(args.timestamps)
    save_dir = "${{outputs.save_dir}}" if args.aml_output_path else args.save_dir
    cmd = (
        f"pip install -e . --quiet && python scripts/ensemble.py"
        f" --input-csv {args.input_csv}"
        f" --model {args.model}"
        f" --save-dir {save_dir}"
        f" --save-folder {args.save_folder}"
        f" --timestamps {timestamps}"
        f" --workers {args.workers}"
        f" --shard-index {shard_index}"
        f" --num-shards {args.num_shards}"
    )
    if args.skip_existing:
        cmd += " --skip-existing"
    if args.no_water:
        cmd += " --no-water"
    if args.no_elevation:
        cmd += " --no-elevation"
    if args.stac_fallback:
        cmd += " --stac-fallback"
    if args.label_url_template:
        cmd += f" --label-url-template '{args.label_url_template}'"
    if args.udm_url_template:
        cmd += f" --udm-url-template '{args.udm_url_template}'"
    if args.water_url_template:
        cmd += f" --water-url-template '{args.water_url_template}'"
    if args.elevation_url_template:
        cmd += f" --elevation-url-template '{args.elevation_url_template}'"
    if args.confidence_threshold != 95:
        cmd += f" --confidence-threshold {args.confidence_threshold}"
    if args.averaging_algorithm != "mean":
        cmd += f" --averaging-algorithm {args.averaging_algorithm}"
    if args.default_behavior != "median":
        cmd += f" --default-behavior {args.default_behavior}"
    if args.elevation_threshold != 5100:
        cmd += f" --elevation-threshold {args.elevation_threshold}"
    if args.small_height_threshold != 0.024:
        cmd += f" --small-height-threshold {args.small_height_threshold}"
    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    ml_client = _get_ml_client(args)
    compute   = args.compute or os.environ["AML_COMPUTE_NAME"]
    env_name  = args.aml_environment or os.environ["AML_ENVIRONMENT_NAME"]

    # SAS tokens are injected as job env vars so they never appear in the command string.
    env_vars = {
        "OUTPUT_SAS": os.environ["OUTPUT_SAS"],
        "UDM_SAS":    os.environ["UDM_SAS"],
    }
    if os.environ.get("BLOB_SAS"):
        env_vars["BLOB_SAS"] = os.environ["BLOB_SAS"]

    outputs = {}
    if args.aml_output_path:
        outputs["save_dir"] = Output(
            type="uri_folder",
            path=args.aml_output_path,
            mode="rw_mount",
        )

    submitted = []
    for shard_index in range(args.num_shards):
        job = command(
            code=str(Path(__file__).parent.parent),  # repo root — uploaded as job source
            command=_build_command(shard_index, args),
            environment=env_name,
            compute=compute,
            display_name=f"ensemble-{args.save_folder}-shard{shard_index:04d}",
            experiment_name=args.experiment_name,
            environment_variables=env_vars,
            outputs=outputs or None,
        )
        created = ml_client.jobs.create_or_update(job)
        submitted.append(created.name)
        logger.info(f"Submitted shard {shard_index}/{args.num_shards}: {created.name}")

    logger.info(f"Submitted {len(submitted)} jobs for experiment '{args.experiment_name}'.")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def set_up_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit ensemble.py as N parallel AML Command jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- AML workspace (falls back to env vars if not supplied) ---
    aml = parser.add_argument_group("AML workspace (override env vars)")
    aml.add_argument("--subscription-id", type=str, default=None)
    aml.add_argument("--resource-group",  type=str, default=None)
    aml.add_argument("--workspace",       type=str, default=None)
    aml.add_argument("--tenant-id",       type=str, default=None,
                     help="Azure tenant ID. Falls back to AZURE_TENANT_ID env var.")
    aml.add_argument(
        "--compute", type=str, default=None,
        help="AML compute target name (falls back to AML_COMPUTE_NAME env var).",
    )
    aml.add_argument(
        "--aml-environment", type=str, default=None,
        help=(
            "Registered AML environment, e.g. 'buildings-ensemble@latest'. "
            "Falls back to AML_ENVIRONMENT_NAME env var."
        ),
    )
    aml.add_argument(
        "--experiment-name", type=str, default="ensemble",
        help="AML experiment name for grouping runs.",
    )

    # --- Sharding ---
    parser.add_argument(
        "--num-shards", type=int, default=119,
        help="Number of parallel AML jobs to submit.",
    )

    # --- Ensemble args (passed through to ensemble.py) ---
    ens = parser.add_argument_group("ensemble.py arguments (passed through to each job)")
    aml.add_argument(
        "--aml-output-path", type=str, default=None,
        help=(
            "AML datastore path to mount as the save directory, e.g. "
            "'azureml://datastores/planet_output/paths/predictions/'. "
            "When set, --save-dir is ignored and outputs go directly to blob storage."
        ),
    )

    ens.add_argument("--input-csv",    required=True,  type=str)
    ens.add_argument("--model",        required=True,  type=str)
    ens.add_argument("--save-dir",     type=str, default="outputs",
                     help="Local output directory. Ignored when --aml-output-path is set.")
    ens.add_argument("--save-folder",  required=True,  type=str)
    ens.add_argument("--timestamps",   required=True,  nargs="+", type=str)
    ens.add_argument("--workers",      type=int,   default=32)
    ens.add_argument("--label-url-template",    type=str, default=None,
                     help="URL template for label TIFFs. Placeholders: {scene}, {timestamp}, {model}, {sas}.")
    ens.add_argument("--udm-url-template",      type=str, default=None,
                     help="URL template for UDM TIFFs. Placeholders: {scene}, {timestamp}, {sas}.")
    ens.add_argument("--water-url-template",    type=str, default=None,
                     help="URL template for pre-computed water rasters. Placeholders: {scene}, {sas}.")
    ens.add_argument("--elevation-url-template", type=str, default=None,
                     help="URL template for pre-computed elevation rasters. Placeholders: {scene}, {sas}.")
    ens.add_argument("--skip-existing",     action="store_true")
    ens.add_argument("--no-water",          action="store_true")
    ens.add_argument("--no-elevation",      action="store_true")
    ens.add_argument("--stac-fallback",     action="store_true",
                     help="Fall back to Planetary Computer STAC if pre-computed rasters are unavailable.")
    ens.add_argument("--confidence-threshold",  type=int,   default=95)
    ens.add_argument("--averaging-algorithm",   type=str,   default="mean",
                     choices=["mean", "max", "min", "mode"])
    ens.add_argument("--default-behavior",      type=str,   default="median",
                     choices=["median", "mean", "max", "min"])
    ens.add_argument("--elevation-threshold",   type=int,   default=5100)
    ens.add_argument("--small-height-threshold", type=float, default=0.024,
                     help="Minimum height value considered a building (band 2).")

    return parser


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
