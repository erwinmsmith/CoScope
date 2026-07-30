# Resuming an Experiment on Another Server

The checkpoint can move without rerunning completed conditions. A safe cutover
has one writer at a time: stop the source, export a consistent bundle, verify
the target, then start the target. Never run both services from copies of the
same checkpoint.

The migration bundle contains the SQLite checkpoint, status metadata, hashes
for all selected benchmark files, the required Git revision, and non-secret
`COSCOPE_*` settings. It never contains API-key values or benchmark data.

## 1. Prepare the target

Clone the repository, install the environment, transfer `/var/lib/coscope/data/`
separately, and create `/etc/coscope/coscope.env` with the same non-secret
settings and valid provider keys. Do not start the service.

The target must check out the exact `code_revision` recorded by the running
experiment—not the newest branch head:

```bash
git -C /opt/coscope checkout <recorded-code-revision>
```

If that revision predates this helper, copy the standalone helper from a
current control checkout to both servers without changing `/opt/coscope`:

```bash
scp coscope/scripts/migrate_cloud_run.py \
  root@HOST:/usr/local/sbin/coscope-migrate-cloud-run.py
ssh root@HOST chmod 700 /usr/local/sbin/coscope-migrate-cloud-run.py
```

## 2. Stop and export the source

```bash
systemctl stop coscope-factorial

/opt/coscope/.venv/bin/python \
  /usr/local/sbin/coscope-migrate-cloud-run.py export \
  --state-dir /var/lib/coscope/runs/factorial-v1 \
  --data-root /var/lib/coscope/data \
  --env-file /etc/coscope/coscope.env \
  --repo-root /opt/coscope \
  --bundle-dir /var/lib/coscope/migrations/factorial-v1-cutover
```

Export refuses an active coordinator or any `running` row. The SQLite backup
also passes `PRAGMA integrity_check`.

## 3. Transfer, verify, and restore

Transfer the bundle and data with `rsync -a --checksum`. On the target:

```bash
/opt/coscope/.venv/bin/python \
  /usr/local/sbin/coscope-migrate-cloud-run.py restore \
  --bundle-dir /var/lib/coscope/migrations/factorial-v1-cutover \
  --data-root /var/lib/coscope/data \
  --env-file /etc/coscope/coscope.env \
  --repo-root /opt/coscope \
  --state-dir /var/lib/coscope/runs/factorial-v1
```

Restore refuses a nonempty target state directory and verifies code, data,
checkpoint, and non-secret runtime configuration before copying anything.

Install the service, adjust only resource controls such as `--workers`, then
start it. Confirm that the target's succeeded count is at least the source's
final count and that failures remain zero.

Keep the stopped source state until the target has checkpointed several new
tasks. If target validation fails, fix it or restart the unchanged source;
never start both.
