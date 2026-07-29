# Aliyun Experiment Deployment

The Git repository contains code and configuration templates only. Keep API
keys in `/etc/coscope/coscope.env`, benchmark files under
`/var/lib/coscope/data/`, and checkpoints under `/var/lib/coscope/runs/`.
These locations must never be copied back into the repository.

## Install

```bash
git clone https://github.com/erwinmsmith/CoScope.git /opt/coscope
cd /opt/coscope
# Before the deployment PR is merged, switch to its published branch.
# git switch agent/cloud-resumable-factorial
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,bench]"

install -d -m 700 /etc/coscope
install -d -m 700 \
  /var/lib/coscope/data /var/lib/coscope/models /var/lib/coscope/runs
cp .env.example /etc/coscope/coscope.env
chmod 600 /etc/coscope/coscope.env
```

Edit `/etc/coscope/coscope.env` on the server and set the provider keys.
Do not use `git add -f` on that file. Transfer datasets separately, for
example:

```bash
rsync -av --partial raw/ root@ALI_HOST:/var/lib/coscope/data/
```

Configure the local model cache as
`/var/lib/coscope/models/fastembed`, initially with
`COSCOPE_EMBEDDING_LOCAL_FILES_ONLY=false`. Download and validate the model:

```bash
.venv/bin/python -m coscope.scripts.check_providers \
  --env-file /etc/coscope/coscope.env
```

After this succeeds, set `COSCOPE_EMBEDDING_LOCAL_FILES_ONLY=true`. The
experiment records the ONNX file SHA-256 in its model identity and shares one
loaded session across all workers. The default 4,096-entry result cache is
bounded and only avoids deterministic re-embedding across factorial arms;
embedding time is separately recorded and excluded from the canonical
end-to-end comparison metric.

Run the full no-provider preflight before starting the service:

```bash
cd /opt/coscope
.venv/bin/python -m coscope.scripts.preflight_experiment \
  --workflow factorial --full --data-root /var/lib/coscope/data \
  --output /var/lib/coscope/runs/full-preflight.json
```

Install and launch the checkpointed runner:

```bash
install -m 644 deploy/ali-root/coscope-factorial.service \
  /etc/systemd/system/coscope-factorial.service
systemctl daemon-reload
systemctl enable --now coscope-factorial
```

The service runs three task workers. Tune `--workers` conservatively according
to provider rate limits; it controls concurrent example-condition executions,
not the internal ToT branch count.

## Progress and recovery

```bash
systemctl status coscope-factorial
journalctl -u coscope-factorial -f

cd /opt/coscope
.venv/bin/python -m coscope.scripts.experiment_status \
  /var/lib/coscope/runs/factorial-local-bge-v1
```

`checkpoint.sqlite3` is the source of truth. `status.json` is replaced
atomically after completions, and `events.jsonl` is append-only. A normal
restart reclaims interrupted `running` rows and never claims `succeeded`
rows. Failed tasks use exponential backoff and stop after three attempts.
After fixing a provider problem, restart once with
`--retry-final-failures`.

Provider APIs do not expose a transaction shared with the local checkpoint.
Therefore a process killed after a provider accepted a request but before the
result was committed may repeat that one in-flight condition. Completed
conditions are exactly-once from the runner's perspective; interruption risk
is bounded to the active `--workers` conditions.

The installed service enables `--purge-details-after-success`. During the run,
non-code tasks retain only compact aggregate inputs; MBPP predictions remain
until EvalPlus finishes. After `final_metrics.json` and `COMPLETED.json` are
atomically committed, the runner deletes SQLite, WAL, event, and detailed
result files. The completion marker prevents a later service restart from
recreating and rerunning the experiment.

Stop cleanly with `systemctl stop coscope-factorial`; active requests finish
and checkpoint before exit. Do not deploy a different Git revision into the
same state directory. Use a new run directory for changed code or experiment
options, because the manifest fingerprint deliberately rejects mixed runs.

For a server cutover, follow
[`docs/CLOUD_MIGRATION.md`](../../docs/CLOUD_MIGRATION.md). The migration
command creates an integrity-checked SQLite snapshot and verifies the exact
Git revision, dataset hashes, and non-secret runtime settings on the target.
It intentionally excludes provider keys and dataset contents. Stop the source
before export, and do not start it again after the target begins writing.

Without the purge option, detailed completed records remain compressed in
SQLite and can be exported when needed:

```bash
.venv/bin/python -m coscope.scripts.experiment_status \
  /var/lib/coscope/runs/factorial-local-bge-v1 \
  --export-results \
  /var/lib/coscope/runs/factorial-local-bge-v1/results.jsonl
```

MBPP-Plus predictions are scored in six checkpointed EvalPlus jobs after
generation finishes. Docker execution remains network-disabled and separate
from the LLM task checkpoints.
