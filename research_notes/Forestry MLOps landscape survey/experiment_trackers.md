# Experiment-tracking and model-registry tools, 2026 state (build-versus-buy input for forestry)

Research date: 2026-10-02. All facts below carry a source URL. Dates pulled from GitHub release pages often omit the year; where the year was inferred or two sources disagreed, the entry says so. Health signals (stars, last push) are GitHub API reads on 2026-10-02 unless noted; download counts are pypistats.org 30-day windows read the same day.

## Per-tool profiles (storage, server need, registry, comparison, provenance, license, health, pain points)

### Takeaway
The field split in 2025-2026: the two big SaaS players changed hands (W&B to CoreWeave, May 2025; Neptune to OpenAI, with the Neptune service deleted on 2026-03-05), MLflow 3.x pivoted to GenAI observability and made SQLite its default while putting the file store into maintenance mode, and a new "lite" tier appeared (Trackio, SQLite per project, 297k downloads/month; SwanLab; mltraq; WaddleML on DuckDB). Aim, Guild AI, Sacred and Omniboard are dormant or slow-moving.

### Cited Findings

**MLflow (Apache-2.0, LF Projects)**
- Latest release 3.16.0 on 2026-09-03; 2026 cadence roughly monthly (3.9.0 Jan 30, 3.10.0 Feb 23, 3.11.1 Apr 7, 3.12.0 May 5, 3.13.0 May 29, 3.14.0 Jun 17, 3.15.0 Jul 31). Headline features are GenAI-side: AI Assistant, judges, trace explorer, MCP registry, RBAC, multi-workspace. MLflow 3 launched 2025-06-11. — [MLflow release archive](https://mlflow.org/releases/archive/)
- Backend stores: "MLflow supports different databases through SQLAlchemy, including sqlite, postgresql, mysql, and mssql." "SQLite is the default backend store. When you start MLflow without specifying a backend, it automatically creates and uses sqlite:///mlflow.db." The file-system backend is "in maintenance mode and will not receive further updates." "Model Registry functionality requires a database-backed store." Schema upgrades: run `mlflow db upgrade [db_uri]`; "Schema migrations can result in database downtime ... Always backup your database before running migrations." — [Backend store architecture](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)
- 3.7.0 release note: "[Tracking] SQLite is now the default backend for the MLflow Tracking server" (#18497). — [v3.7.0 release](https://github.com/mlflow/mlflow/releases/tag/v3.7.0)
- 3.13.0 release note: "Pointing the tracking or model registry store at a local file-system path now raises an error by default; set MLFLOW_ALLOW_FILE_STORE=true to keep using a file-based store." — [v3.13.0 release](https://github.com/mlflow/mlflow/releases/tag/v3.13.0)
- Troubleshooting doc: default file-based store "severely limits the performance, e.g., no indexing"; recommends `mlflow server --backend-store-uri sqlite:///mlflow.db`; async logging exists for slow `log_metric`; runs and models are logically deleted, `mlflow gc` purges; "SDK and the server within the same major version are expected to work together." — [Self-hosting troubleshooting](https://mlflow.org/docs/latest/self-hosting/troubleshooting/)
- UI slowness is a known, closed issue: loading 100 more rows went from 3 s at 100 runs to 59 s at 1000 runs; root cause ag-grid rendering, fixed via PR #5725. — [mlflow#5653](https://github.com/mlflow/mlflow/issues/5653)
- System metrics (opt-in): `system/cpu_utilization_percentage`, `system/system_memory_usage_megabytes`, `system/gpu_utilization_percentage`, `system/gpu_memory_usage_megabytes`, `system/gpu_power_usage_watts`, network and disk counters. Needs `psutil`; `nvidia-ml-py` for NVIDIA; `pyrsmi` for AMD. Enable via `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING`, `mlflow.enable_system_metrics_logging()`, or `log_system_metrics=True`. The doc lists no hardware-identity fields (no CPU model, no GPU name). — [System metrics](https://mlflow.org/docs/latest/ml/tracking/system-metrics/)
- Git provenance tags set automatically: `mlflow.source.name`, `mlflow.source.type`, `mlflow.source.git.commit`, `mlflow.source.git.branch`, `mlflow.source.git.repoURL`. — [Track environments and context](https://mlflow.org/docs/latest/genai/tracing/track-environments-context/)
- Registry stores registered models, auto-incremented versions ("When a new model is added to the Model Registry, it is added as version 1"), mutable aliases such as `@champion`, tags at model and version level, lineage to the source run, markdown descriptions; `mlflow.create_external_model()` since MLflow 3. — [Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- `mlflow.search_runs`: "The output will be a pandas DataFrame with the runs that match the specified filters"; SQL-like filter DSL with `=`, `!=`, `<`, `LIKE`, `ILIKE`, `IN`, `AND` (no `OR`). — [Search runs](https://mlflow.org/docs/latest/ml/search/search-runs/)
- Telemetry: "Starting with version 3.2.0, MLflow collects anonymized usage data by default"; opt out with `MLFLOW_DISABLE_TELEMETRY=true`. — [Usage tracking](https://mlflow.org/docs/latest/community/usage-tracking/)
- Health: 21.36M downloads in the last 30 days, latest 3.16.1. — [pypistats mlflow](https://pypistats.org/packages/mlflow)
- Independent cost framing (Aug 2026 article): self-hosted MLflow (tracking server + Postgres + object store) "typically lands well under $100/month"; MLflow gaps named: basic UI, no native alerting, no built-in sweep orchestration. — [Spheron, W&B pricing vs self-hosted MLflow 2026](https://www.spheron.network/blog/weights-biases-pricing-vs-self-hosted-mlflow-2026/)

**Weights & Biases (client MIT; server proprietary license key; owned by CoreWeave)**
- CoreWeave completed the acquisition on 2025-05-05. — [CoreWeave press release](https://coreweave.com/news/coreweave-completes-acquisition-of-weights-biases-2); deal value $1.7B. — [CoreWeave investor release](https://investors.coreweave.com/news/news-details/2025/CoreWeave-to-Acquire-Weights--Biases---Industry-Leading-AI-Developer-Platform-for-Building-and-Deploying-AI-Applications/default.aspx)
- As of 2026-10-02, `docs.wandb.ai` 301-redirects to `docs.coreweave.com`, and `wandb.ai/site/pricing` redirects to `coreweave.com/forge-pricing`; several docs.coreweave.com pages returned an "Enter access code" wall during this research (a public-docs regression or gating; see Gaps). — [Forge pricing page](https://coreweave.com/forge-pricing)
- Pricing (Forge page, which lists "Weights & Biases Models" features): Free $0, up to 5 model seats, 5 GB storage, 1 GB/month Weave ingestion; Pro "Starts at $60/month", up to 10 seats, 100 GB (then $0.03/GB), 1.5 GB Weave; Enterprise custom, bring-your-own-bucket option; Academic free, 200 GB, up to 100 seats. — [Forge pricing](https://coreweave.com/forge-pricing)
- Offline: set `WANDB_MODE=offline`; later `wandb sync [RUN-DIRECTORY]`; check `run.settings._offline`. — [How do I run wandb offline](https://docs.coreweave.com/support/models/articles/how-do-i-run-wandb-offline.md); CLI `wandb offline` / `wandb online` / `wandb disabled` — [wandb offline CLI](https://docs.wandb.ai/models/ref/cli/wandb-offline.md), [wandb disabled CLI](https://docs.wandb.ai/models/ref/cli/wandb-disabled.md); `WANDB_DISABLED=true`, `WANDB_DIR` controls where generated files land. — [Environment variables](https://docs.wandb.ai/guides/track/environment-variables)
- Self-hosted W&B Server: `pip install wandb && wandb server start`, or `docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local`; the container bundles MySQL and MinIO; a free license is generated at deploy.wandb.ai and pasted into `/system-admin`; production wants external MySQL 8, S3 and Redis, with a paid license ("contact@wandb.com"). Repo is 359 stars, pushed 2026-10-01. — [wandb/server README](https://github.com/wandb/server), [GitHub API wandb/server](https://api.github.com/repos/wandb/server)
- Client: 11,269 stars, MIT, pushed 2026-10-03, 1,003 open issues. — [GitHub API wandb/wandb](https://api.github.com/repos/wandb/wandb); 13.71M downloads/30 days, latest 0.30.0. — [pypistats wandb](https://pypistats.org/packages/wandb)
- Provenance: `wandb.init` "automatically collects Git information, including the remote repository link and the SHA of the latest commit." — [Save the git commit](https://docs.wandb.ai/support/models/articles/how-can-i-save-the-git-commit-associated.md). The per-run `wandb-metadata.json` holds OS string, Python version, hostname, GPU name and count, physical and logical CPU counts, git remote and commit. — [Kempner Institute handbook, reproducing W&B runs](https://handbook.eng.kempnerinstitute.harvard.edu/s5_ai_scaling_and_engineering/experiment_management/reproducing_runs.html)
- System-metrics collection can be turned off with `wandb.init(settings=wandb.Settings(x_disable_stats=True))`. — [Disable system metrics](https://docs.wandb.ai/support/how_can_i_disable_logging_of_system_metrics_to_wb)
- Pain points from the community forum: `wandb sync` retries deleted runs about seven times per folder before moving on; ~300 MB artifacts can take hours to sync; users at ~2k runs/day went offline to dodge network strain and hit errors; upload time blocks sequential runs. — [Sync offline runs thread](https://community.wandb.ai/t/sync-local-offline-runs-to-the-dashboard-while-deleting-old-folders/4786), [Many quick runs thread](https://community.wandb.ai/t/best-practices-for-many-quick-runs/1145), [Sync extremely slow](https://lightrun.com/answers/wandb-wandb-sync-with-server-is-extremely-slow), [Silently dead syncer (Ray forum)](https://discuss.ray.io/t/help-investigating-wandb-offline-experiments-sometimes-not-synced-silently-dead-syncer/23041)
- Independent pricing read (Aug 2026): Enterprise quotes of $315 to $400 per seat per month, median annual contract $47,625 (Vendr data); self-hosting is "Enterprise-only" in the author's framing. — [Spheron 2026 article](https://www.spheron.network/blog/weights-biases-pricing-vs-self-hosted-mlflow-2026/)

**Neptune (gone)**
- "Neptune has been acquired by OpenAI"; transition window "December 3, 2025 → March 5, 2026"; SaaS shut down "March 5, 2026, at 10 am PST"; remaining hosted data "securely and irreversibly deleted"; Helm/container image repos deleted 2026-03-08; GitHub repos "Switched to archived (read-only)"; the Neptune Exporter worked until the shutdown. Recommended destinations: Comet, GoodSeed, Lightning AI, Minfx.ai, MLflow, Pluto, Weights & Biases, ZenML. — [Neptune transition hub](https://docs.neptune.ai/transition_hub)
- Support article: discontinued 2026-03-06 12:00 UTC; "no export, no restore, and no recovery path." — [Service shutdown overview](https://support.neptune.ai/en/articles/13925165-service-shutdown-overview)
- `neptune-client` is archived, Apache-2.0, 623 stars, last push 2026-03-17. — [GitHub API neptune-client](https://api.github.com/repos/neptune-ai/neptune-client)
- Reported price under $400M in stock. — [VKTR](https://www.vktr.com/ai-news/openai-buys-neptune-for-under-400m-in-ai-governance-push/), [Notes from Poland](https://notesfrompoland.com/2025/12/04/openai-to-acquire-polish-founded-startup/)
- Pre-shutdown pricing (secondary aggregator): Lab tier $250/user/month, Self-Hosted "Contact Sales". — [Costbench](https://www.costbench.com/software/mlops/neptune-ai/)

**Comet (SDK MIT on PyPI; platform proprietary; Opik open source)**
- Offline mode: `comet_ml.start(online=False)`; data saved as a ZIP such as `.cometml-runs/<id>.zip`; upload with `comet upload <zip>`; "other than the offline logging, the experiment will behave exactly the same as an online experiment." — [Run experiments offline](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/advanced/running-offline/)
- Pricing (MLOps platform): Free $0, "Fair usage policy", 100 GB, 1 user; Pro $19/user/month, 1500 training hours, 500 GB then $3/100GB/month, up to 10 users; Enterprise custom. Opik: Free up to 10 members, 25k spans/month, 60-day retention; Pro $19/month; open-source self-host free. "Both platforms support self-hosted deployment options." — [Comet pricing](https://www.comet.com/site/pricing/)
- Auto-logged on `Experiment` start: metrics, hyperparameters, the code file, "the git commit and git patch (uncommitted files)", console output, installed Python packages, system metrics (memory, GPU). `ExperimentConfig` exposes `log_env_gpu` (GPU details and metrics), `log_env_host` (ip, hostname, python version, user), `log_env_cpu`. — [Create an experiment](https://comet.com/docs/v2/guides/experiment-management/create-an-experiment/), [ExperimentConfig reference](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/ExperimentConfig/)
- Health: `comet-ml` 3.58.7, 346,851 downloads/30 days, MIT, depends on `sentry-sdk`. — [pypistats comet-ml](https://pypistats.org/packages/comet-ml)
- Opik self-host docs exist (Kubernetes "production ready"). — [Opik self-host overview](https://www.comet.com/docs/opik/self-host/overview)

**ClearML (SDK Apache-2.0; server SSPL-1.0)**
- Server is multi-container: Elasticsearch, MongoDB, Redis, API server (8008), web server (8080), file server (8081), agent-services, async_delete. "Deploying the server requires a minimum of 8 GB of memory, 16 GB is recommended." Elasticsearch needs `vm.max_map_count` 524288. Backups require stopping the server. — [ClearML Server on Linux/macOS](http://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_linux_mac/)
- `clearml-server` license: Server Side Public License v1.0; 470 stars. — [clearml/clearml-server](https://github.com/clearml/clearml-server)
- Offline: `Task.set_offline(offline_mode=True)` or `CLEARML_OFFLINE_MODE=1`; data zipped at `~/.clearml/cache/offline/<task_id>.zip`; import with `clearml-task --import-offline-session` or `Task.import_offline_session()`. "Offline mode only works with tasks created using Task.init() and not with those created using Task.create()." — [Set offline](http://clear.ml/docs/latest/docs/guides/set_offline/)
- Hosted pricing: Community free, up to 3 users, 100 GB artifacts, 1 GB metric events, 1M API calls/month; Pro $15/user/month + usage, up to 10 users; Scale/Enterprise quoted. Open-source server excludes Hyper-Datasets, dynamic resource allocation, multi-tenancy, configuration vault, LDAP/RBAC. — [ClearML pricing](https://clear.ml/pricing/)
- Auto-capture: "Full source control info, including non-committed local changes", packages and versions, "Resource Monitoring (CPU/GPU utilization, temperature, IO, network, etc.)", hostname, stdout/stderr, model snapshots. — [clearml on PyPI](https://pypi.org/project/clearml/)
- CLI: `clearml-task` creates and enqueues tasks from the command line. — [ClearML Task CLI](https://clear.ml/docs/latest/docs/apps/clearml_task/)
- Health: 709,013 downloads/30 days. — [pypistats clearml](https://pypistats.org/packages/clearml)

**Aim (Apache-2.0, AimStack)**
- Self-hosted, ".aim directory" repo, `aim up` starts the UI server, remote tracking server for multi-host, system resource tracking, Python-expression query SDK, 6.3k stars on the README. — [aimhubio/aim](https://github.com/aimhubio/aim)
- Storage is a collection of RocksDB databases; unindexed run chunks slow queries; `aim up` spawns a background reindex thread; soft file locks guard against corruption. — [Storage indexing](https://aimstack.readthedocs.io/en/latest/understanding/storage_indexing.html), [Data storage](https://aimstack.readthedocs.io/en/latest/understanding/data_storage.html)
- Latest release v3.29.1. The GitHub tag page reads "08 May" and the PyPI page reports May 8, 2025; the releases list fetch rendered it as 2024 (year conflict, see Gaps). Preceding releases: 3.28.0 Mar 21, 3.27.0 Dec 18, 3.26.1 Dec 3, 3.25.1 Nov 6 (2023-2024). — [Aim releases](https://github.com/aimhubio/aim/releases), [v3.29.1 tag](https://github.com/aimhubio/aim/releases/tag/v3.29.1), [aim on PyPI](https://pypi.org/project/aim/)
- Repo still receives pushes (2026-10-02) but no release since 3.29.1; 6,274 stars; 480 open issues; Python >=3.7. — [GitHub API aim](https://api.github.com/repos/aimhubio/aim), [aim on PyPI](https://pypi.org/project/aim/)
- 82,850 downloads/30 days. — [pypistats aim](https://pypistats.org/packages/aim)
- Pain: "aim is very slow since 3.12.0", `aim up` near a minute, `aim.Run` creation 1-2 minutes reported. — [Lightrun summary of aim slowness](https://lightrun.com/answers/aimhubio-aim-aim-is-very-slow-since-3120), [aim#2999](https://github.com/aimhubio/aim/issues/2999)

**Trackio (MIT, Hugging Face / Gradio team)**
- "A lightweight, free experiment tracking Python library built on top of Hugging Face Buckets and Spaces"; API compatible with `wandb.init`, `wandb.log`, `wandb.finish` (`import trackio as wandb`); local-first dashboard; optional `space_id` (HF Space) or `server_url` (self-hosted Trackio server); core "<3,000 lines of Python"; "LLM-friendly: Designed for autonomous ML experiments with CLI commands and Python APIs". Dataset persistence (`dataset_id`) is deprecated in favor of Buckets. — [Trackio docs index](https://huggingface.co/docs/trackio/index)
- Storage: one SQLite file per project at `TRACKIO_DIR` (default `~/.cache/huggingface/trackio/{project}.db`); media under `TRACKIO_DIR/media/`; tables `metrics`, `configs`, `system_metrics`, `project_metadata`, `pending_uploads`, `alerts`, `artifacts`, `artifact_versions` (sequential version, SHA-256 manifest digest, `producer_run_id`), `artifact_aliases` (e.g. `latest`), `run_artifact_links` (input/output). Metrics and configs are JSON text blobs. Parquet export flattens JSON to columns (`{project}.parquet`, `_system`, `_configs`, `_traces`, artifact tables). `trackio query project --project X --sql "SELECT ..." [--json]` runs read-only SQL locally or against a Space. "Trackio is still in beta ... future releases may evolve the schema and require migrations or regenerated local databases." — [Storage schema and direct queries](https://huggingface.co/docs/trackio/v0.32.2/storage_schema)
- System metrics: `pip install trackio[gpu]` (NVIDIA) or `trackio[apple-gpu]`; logged every 10 s by default, `gpu_log_interval` and `auto_log_gpu` control it; per-GPU utilization, memory, temperature, power. — [Trackio track docs](https://huggingface.co/docs/trackio/v0.32.2/track)
- Public API: `init`, `log`, `finish`, `show(project, theme, mcp_server)` (the dashboard can run as an MCP server), `import_csv`, `import_tf_events`, `Table`, `Image`, `Video`. — [API reference](https://huggingface.co/docs/trackio/main/api)
- Health: 1,703 stars, created 2025-05-08, pushed 2026-09-30, 1 open issue; releases 0.40.0 (Sep 30), 0.39.0 (Sep 24, adds artifact overwrite), 0.38.1 (Sep 17). — [GitHub API trackio](https://api.github.com/repos/gradio-app/trackio), [Trackio releases](https://github.com/gradio-app/trackio/releases)
- 297,035 downloads/30 days; Python >=3.10; deps include gradio-client, huggingface-hub, numpy, orjson, pillow, starlette, uvicorn. — [pypistats trackio](https://pypistats.org/packages/trackio), [PyPI JSON](https://pypi.org/pypi/trackio/json)
- Launch coverage (Sep 2025): under 1,000 lines at launch, SQLite with Parquet backup to HF every five minutes when synced. — [InfoQ](https://www.infoq.com/news/2025/09/hugging-face-trackio)
- ZenML ships a Trackio experiment-tracker stack component. — [ZenML docs](https://docs.zenml.io/stacks/stack-components/experiment-trackers/trackio)

**DVC + DVCLive, GTO, DagsHub (Apache-2.0)**
- DVCLive logs into a `dvclive` directory "and tracked as a DVC experiment for analysis and comparison"; no server; compare with `dvc exp show` (has `--csv`/`--json`), the VS Code extension, or DVC Studio; `Live.monitor_system()` exists. — [DVCLive docs](https://doc.dvc.org/dvclive)
- `monitor_system()` metrics: `system/cpu/count`, `system/cpu/usage (%)`, `system/ram/usage (GB)`, `system/ram/total (GB)`, `system/disk/usage (GB)/<name>`, `system/gpu/count`, `system/gpu/usage (%)/<idx>`, `system/vram/usage (GB)/<idx>`, `system/vram/total (GB)/<idx>`; no GPU name field documented. — [monitor_system](https://doc.dvc.org/dvclive/live/monitor_system)
- Experiments are "custom Git references (found in .git/refs/exps) with one or more commits based on HEAD"; not pushed by default; `dvc exp push` shares them. — [Experiments overview](https://doc.dvc.org/user-guide/experiment-management/experiments-overview)
- GTO: "Turn your Git repository into an Artifact Registry or Model Registry"; versions and stages stored as Git annotated tags; 161 stars; Apache-2.0; pushed 2026-10-03; 37 open issues. — [iterative/gto](https://github.com/iterative/gto), [GitHub API gto](https://api.github.com/repos/iterative/gto)
- DVCLive 3.49.1, Python >=3.9, requires `dvc>=3.48.4`; 99,838 downloads/30 days. — [PyPI JSON dvclive](https://pypi.org/pypi/dvclive/json), [pypistats dvclive](https://pypistats.org/packages/dvclive)
- DagsHub is operating: Individual free (20 GB, unlimited public repos), Team $99-$119/user/month, Enterprise with on-prem; experiment tracking "Compatible with MLflow". — [dagshub.com](https://dagshub.com/), [DagsHub experiment docs](https://dagshub.com/docs/use_cases/track_ml_experiments/)

**Guild AI (Apache-2.0)**
- Runs unmodified scripts, stores each run in a runs directory, `guild view` UI, source snapshot per run. 909 stars. — [guildai/guildai](https://github.com/guildai/guildai)
- Last push 2025-04-29; 241 open issues; not archived. — [GitHub API guildai](https://api.github.com/repos/guildai/guildai)
- Releases page shows 0.8.1 (May 11, year not rendered) as newest; PyPI latest is 0.9.0; 529 downloads/30 days. — [Guild releases](https://github.com/guildai/guildai/releases), [pypistats guildai](https://pypistats.org/packages/guildai)

**Sacred + Omniboard (MIT)**
- Sacred 0.8.7 released 2024-11-26, Python >=3.8. — [PyPI JSON sacred](https://pypi.org/pypi/sacred/json); repo 4,381 stars, pushed 2025-10-22, 108 open issues. — [GitHub API sacred](https://api.github.com/repos/IDSIA/sacred); 22,175 downloads/30 days. — [pypistats sacred](https://pypistats.org/packages/sacred)
- Collected automatically: host `cpu`, `hostname`, `os`, `python_version`, `gpu`, optional ENV vars; experiment sources, dependencies as `package==version`, git repo URL, commit, dirty flag; run record with start/stop/heartbeat, status, fail_trace, result. — [Sacred collected information](https://sacred.readthedocs.io/en/stable/collected_information.html)
- Observers: MongoDB (primary), FileStorage, TinyDB, S3, Neptune (dead target). — [sacred on PyPI](https://pypi.org/project/sacred/), [IDSIA/sacred](https://github.com/IDSIA/sacred)
- Omniboard: React/Node dashboard over MongoDB >=4.0; 549 stars; last release 2.16.1 on 2021-12-26; last push 2023-02-01. — [omniboard](https://github.com/vivekratnavel/omniboard), [Omniboard releases](https://github.com/vivekratnavel/omniboard/releases), [GitHub API omniboard](https://api.github.com/repos/vivekratnavel/omniboard)

**TensorBoard (Apache-2.0, Google)**
- Latest 2.21.0; 2.19.0 uploaded 2025-02-12 (PyPI). GitHub release pages: 2.19.0 Feb 12, 2.20.0 Jul 17 (Python 3.13 support), 2.21.0 Jun 29 (time-series tooltip tweaks, Projector CVE fix). Year rendering was inconsistent; see Gaps. — [TensorBoard releases](https://github.com/tensorflow/tensorboard/releases), [PyPI JSON tensorboard](https://pypi.org/pypi/tensorboard/json)
- TensorBoard.dev shut down 2024-01-01; `tensorboard dev upload` removed; local TensorBoard unaffected. — [tensorboard.dev notice](https://tensorboard.dev/)
- HParams plugin caveat: hyperparameters can go missing from the table when one server mixes experiments with differing hparam sets. — [Bits of experience (practitioner blog)](https://audiolabs-erlangen.com/fau/assistant/gaznepoglu/bits_of_experience)
- W&B's own positioning against TensorBoard: code-version capture and run organization as the differentiators. — [How is W&B different from TensorBoard](https://docs.wandb.ai/support/models/articles/how-is-wb-different-from-tensorboard)

**Lightning / Fabric loggers (Apache-2.0)**
- Fabric ships `Logger` base, `CSVLogger` ("Logs metrics to CSV format files"), `TensorBoardLogger`, and `WandbLogger`. — [Fabric loggers API source](https://pytorch-lightning.readthedocs.io/en/2.6.2/fabric/_sources/api/loggers.rst.txt)
- PyTorch Lightning Trainer loggers: LitLogger, CometLogger, CSVLogger, MLFlowLogger, TensorBoardLogger, WandbLogger; "By default, Lightning uses TensorBoard logger under the hood, and stores the logs to a directory (by default in lightning_logs/)"; multiple loggers can be passed as a list; custom loggers subclass `Logger`. — [Lightning logging docs source](https://pytorch-lightning.readthedocs.io/en/2.6.2/pytorch/_sources/extensions/logging.rst.txt)

**Other lite trackers (2025-2026)**
- SwanLab (Apache-2.0): `mode='local'` disables cloud sync, `swanlab watch ./logs` opens a local dashboard; community self-host supports offline use. — [What is SwanLab](https://docs.swanlab.cn/en/guide_cloud/general/what-is-swanlab.html). Self-host: "CPU >= 2 cores, Memory >= 4GB, Storage space >= 20GB"; docker-compose with gateway (8000), MinIO (9000, fixed), Traefik; one-click `docker/install.sh`; Community Edition needs a free license from the SwanLab site. — [SwanLab Docker deploy](https://docs.swanlab.cn/en/self_host/docker/deploy.html). Hardware monitors: NVIDIA, AMD ROCm, Ascend NPU, Cambricon, Kunlunxin, Moore Threads, Metax, Iluvatar, Hygon; records git repo, Python env, pip list, run directory; `swanlab.Api` and OpenAPI. — [SwanHubX/SwanLab](https://github.com/SwanHubX/SwanLab). 4,245 stars, pushed 2026-09-24; PyPI 0.10.1 uploaded 2026-09-22, Python >=3.9. — [GitHub API SwanLab](https://api.github.com/repos/SwanHubX/SwanLab), [PyPI JSON swanlab](https://pypi.org/pypi/swanlab/json)
- mltraq (BSD-3-Clause): SQLAlchemy 2.0 persistence, SQLite default, "any SQL database supported by SQLAlchemy"; `.persist()`, `.reload()`, `.df()` to pandas; no UI; 43 stars, last push 2025-03-10. — [mltraq.com](https://www.mltraq.com/), [GitHub API mltraq](https://api.github.com/repos/elehcimd/mltraq)
- WaddleML (MIT): "Lightweight ML tracking & visualization with DuckDB"; 6 stars; pushed 2026-04-06. — [GitHub API waddleml](https://api.github.com/repos/briangu/waddleml)
- experiment-results-manager: "light-weight alternative to mlflow ... that doesn't require kubernetes". — [PyPI ERM](https://pypi.org/project/experiment-results-manager)

### Inferences
- MLflow is the only general-purpose tracker whose default now matches forestry's constraint (one SQLite file, no server) while also carrying a registry, pandas export and git tags, but its 2026 release energy is in GenAI tracing and the registry still requires that database (no git- or file-backed registry).
- Trackio is the closest existing "lite" shape to what forestry describes (SQLite per project, read-only SQL CLI, artifact versions with digests and aliases, agent-oriented API), with the explicit caveat that the schema is beta.
- Neptune's deletion and W&B's docs moving behind CoreWeave are the two concrete lock-in events of 2025-2026; both argue for a tracker whose on-disk format is plain SQLite/Parquet.

### Gaps
- WebFetch summaries of GitHub release pages frequently dropped or mis-inferred the year (Aim v3.29.1: 2024 vs 2025; TensorBoard 2.21.0 vs 2.20.0 ordering; Trackio 0.40.0 rendered as 2024 despite the repo being created 2025-05-08). Treat ISO timestamps from the GitHub API and PyPI JSON as authoritative and the release-page years as approximate.
- Comet on-prem hardware requirements and whether on-prem is Enterprise-only: the deployment docs returned 404 and a public mirror returned 403.
- W&B self-managed "basic setup" docs and the system-metrics reference pages were behind an access-code wall on docs.coreweave.com; the wandb/server README and a third-party handbook were used instead.
- No formal announcement of a "Forge" rebrand was found; the redirect from wandb.ai/site/pricing to coreweave.com/forge-pricing is the only evidence.
- GoodSeed, Minfx.ai and Pluto (named by Neptune as migration targets) returned no indexed material; unassessed.

## Which tools work fully offline/local-first with a file or SQLite backend, and what do you lose versus the server mode?

### Takeaway
Truly serverless with a local store: MLflow (SQLite default, in-process `mlflow` client with `mlflow ui` only for viewing), Trackio (SQLite per project, local Gradio dashboard), DVCLive (files plus git refs), TensorBoard and Lightning CSVLogger (plain files), Sacred FileStorage, Aim (RocksDB dir, but the UI is a server process), mltraq and WaddleML (SQLite/DuckDB, no UI). W&B, Comet and ClearML offer offline capture that is a buffer to be synced later; you lose the UI, comparison and registry until you upload.

### Cited Findings
- MLflow: SQLite is the default store, file store is maintenance-mode and errors unless `MLFLOW_ALLOW_FILE_STORE=true` (3.13.0); registry requires a database store. — [Backend store](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/), [v3.13.0](https://github.com/mlflow/mlflow/releases/tag/v3.13.0)
- MLflow loses nothing functionally without a server beyond multi-user access; the UI is `mlflow server --backend-store-uri sqlite:///mlflow.db` on demand. — [Troubleshooting](https://mlflow.org/docs/latest/self-hosting/troubleshooting/)
- Trackio: dashboard "runs locally by default"; SQLite files in `~/.cache/huggingface/trackio`; `trackio query` works offline; syncing to a Space or Bucket is optional. — [Trackio index](https://huggingface.co/docs/trackio/index), [Storage schema](https://huggingface.co/docs/trackio/v0.32.2/storage_schema)
- W&B offline: `WANDB_MODE=offline` then `wandb sync`; the doc lists no feature loss, but the UI, reports and registry live only on the server, so nothing is viewable until synced. — [Run wandb offline](https://docs.coreweave.com/support/models/articles/how-do-i-run-wandb-offline.md). Sync is the pain point (retries, hours for large artifacts, silently dead syncers). — [W&B forum](https://community.wandb.ai/t/sync-local-offline-runs-to-the-dashboard-while-deleting-old-folders/4786), [Ray forum](https://discuss.ray.io/t/help-investigating-wandb-offline-experiments-sometimes-not-synced-silently-dead-syncer/23041)
- Comet offline: ZIP per experiment, `comet upload`; "behave exactly the same" during logging; viewing requires the hosted or self-hosted server. — [Running offline](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/advanced/running-offline/)
- ClearML offline: ZIP under `~/.clearml/cache/offline/`, imported later; only `Task.init()` tasks, not `Task.create()`. — [Set offline](http://clear.ml/docs/latest/docs/guides/set_offline/)
- DVCLive: everything is files in `dvclive/` plus git refs under `.git/refs/exps`; no server; `dvc exp show --csv/--json`. — [DVCLive](https://doc.dvc.org/dvclive), [Experiments overview](https://doc.dvc.org/user-guide/experiment-management/experiments-overview)
- Aim: local `.aim` repo of RocksDB databases; `aim up` is a local web server you run yourself; remote tracking server optional. — [Data storage](https://aimstack.readthedocs.io/en/latest/understanding/data_storage.html), [aimhubio/aim](https://github.com/aimhubio/aim)
- Sacred: FileStorage and TinyDB observers need no server; MongoDB observer is what Omniboard requires. — [sacred on PyPI](https://pypi.org/project/sacred/), [omniboard](https://github.com/vivekratnavel/omniboard)
- TensorBoard: local tool "unaffected" by the tensorboard.dev shutdown; event files on disk. — [tensorboard.dev](https://tensorboard.dev/)
- Lightning: `CSVLogger` writes CSV/YAML locally; default `TensorBoardLogger` writes to `lightning_logs/`. — [Fabric loggers](https://pytorch-lightning.readthedocs.io/en/2.6.2/fabric/_sources/api/loggers.rst.txt), [Lightning logging](https://pytorch-lightning.readthedocs.io/en/2.6.2/pytorch/_sources/extensions/logging.rst.txt)
- SwanLab: `mode='local'` plus `swanlab watch` for a local dashboard. — [What is SwanLab](https://docs.swanlab.cn/en/guide_cloud/general/what-is-swanlab.html)
- mltraq: SQLite default, pandas `.df()`, no UI. — [mltraq.com](https://www.mltraq.com/)

### Inferences
- For a laptop plus ephemeral pods, the tools whose offline artifact is a self-describing SQLite/Parquet/CSV file (MLflow SQLite, Trackio, DVCLive, mltraq) can be rsynced off a pod and merged or queried with no vendor process; the ZIP-buffer tools (W&B, Comet, ClearML) need their server to become useful.
- MLflow SQLite from multiple pods means multiple `mlflow.db` files; MLflow has no documented merge tool, so cross-pod consolidation is a DIY step (see Gaps).

### Gaps
- No primary source found on merging two MLflow SQLite stores or two Trackio project databases produced on different hosts; Trackio's `log_id` dedup columns hint at sync support but the doc covers only Space sync.

## Which capture hardware/host provenance natively, and which need custom tags?

### Takeaway
Hardware identity (GPU name, CPU counts, hostname, OS) is captured natively by W&B (`wandb-metadata.json`), Sacred (host info), Comet (`log_env_host`, `log_env_gpu`, `log_env_cpu`), ClearML and SwanLab. MLflow, DVCLive and Trackio log utilization time series only (GPU index, not GPU model); CPU model strings are not documented by any tool checked. Nobody documents container cgroup quotas (CPU shares, memory limit); that needs custom tags everywhere.

### Cited Findings
- W&B `wandb-metadata.json`: OS string, Python version, host, GPU name and count (example "NVIDIA L40S"), physical and logical CPU counts, git remote and commit. — [Kempner handbook](https://handbook.eng.kempnerinstitute.harvard.edu/s5_ai_scaling_and_engineering/experiment_management/reproducing_runs.html); git auto-captured on `wandb.init`. — [W&B git commit article](https://docs.wandb.ai/support/models/articles/how-can-i-save-the-git-commit-associated.md)
- Sacred host info: `cpu`, `hostname`, `os`, `python_version`, `gpu`, optional ENV; git URL, commit, dirty. — [Collected information](https://sacred.readthedocs.io/en/stable/collected_information.html)
- Comet: `log_env_gpu` ("GPU details and metrics"), `log_env_host` ("ip, hostname, python version, user"), `log_env_cpu`; git commit plus patch of uncommitted changes; installed packages. — [ExperimentConfig](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/ExperimentConfig/), [Create an experiment](https://comet.com/docs/v2/guides/experiment-management/create-an-experiment/)
- ClearML: source control incl. uncommitted changes, packages, hostname, CPU/GPU utilization, temperature, IO, network. — [clearml on PyPI](https://pypi.org/project/clearml/)
- SwanLab: nine accelerator families monitored; git repo, Python env, pip list, run directory recorded. — [SwanHubX/SwanLab](https://github.com/SwanHubX/SwanLab)
- MLflow: utilization-only `system/*` metrics; git tags `mlflow.source.git.commit/branch/repoURL`; no hardware identity fields documented. — [System metrics](https://mlflow.org/docs/latest/ml/tracking/system-metrics/), [Context tags](https://mlflow.org/docs/latest/genai/tracing/track-environments-context/)
- DVCLive `monitor_system()`: counts and utilization by index only. — [monitor_system](https://doc.dvc.org/dvclive/live/monitor_system)
- Trackio: per-GPU utilization, memory, temperature, power every 10 s for NVIDIA or Apple silicon; no git or host identity documented. — [Trackio track](https://huggingface.co/docs/trackio/v0.32.2/track), [Trackio index](https://huggingface.co/docs/trackio/index)
- Aim: "system info and resource usage", git info, env vars, CLI args, dependencies (README claim). — [aimhubio/aim](https://github.com/aimhubio/aim)

### Inferences
- For same-pod interleaved benchmarking, the needed provenance (CPU model string, GPU model, driver, container memory/CPU quota, pod id) exceeds what any tool records by default; a small provenance collector writing tags/params is required regardless of tracker, which weakens the "buy" case on this axis.

### Gaps
- No tool's documentation mentions cgroup/container limits or CPU model name; absence in docs is not proof of absence in code (W&B's metadata may include a `cpu` brand string; not verified from a primary W&B page because the docs mirror was gated).

## What does each tool's model registry actually store and version, and can a registry live in git or a flat file?

### Takeaway
Only GTO (DVC ecosystem) and Trackio's artifact tables qualify as git- or flat-file registries: GTO stores versions and stages as git annotated tags; Trackio stores artifact versions, SHA-256 manifest digests, aliases and run links in the project SQLite file. MLflow's registry (names, auto-incremented versions, aliases, tags, lineage) requires a database store, which in practice can be the local SQLite file. W&B, Comet and ClearML registries live on their servers.

### Cited Findings
- MLflow registry entities: registered model, sequential versions, mutable aliases (`@champion`), tags at both levels, lineage to run, markdown descriptions; `create_external_model()` in MLflow 3. — [Model Registry](https://mlflow.org/docs/latest/ml/model-registry/). "Model Registry functionality requires a database-backed store." — [Backend store](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/). Registry pointed at a file path errors unless `MLFLOW_ALLOW_FILE_STORE=true`. — [v3.13.0](https://github.com/mlflow/mlflow/releases/tag/v3.13.0)
- GTO: "Turn your Git repository into an Artifact Registry or Model Registry"; version and stage info "using Git annotated tags in a standard format"; GitOps signals. — [iterative/gto](https://github.com/iterative/gto)
- Trackio: `artifacts` (name, type), `artifact_versions` (sequential version from 0, `manifest_digest` SHA-256, manifest entries with path/digest/size, `producer_run_id`), `artifact_aliases` (`latest`), `run_artifact_links` (input/output); static export writes blobs to `artifacts/blobs/sha256/{prefix}/{digest}`. — [Storage schema](https://huggingface.co/docs/trackio/v0.32.2/storage_schema)
- W&B: "asset registry" is a Free-tier feature on the hosted product. — [Forge pricing](https://coreweave.com/forge-pricing)
- ClearML: file server "stores media and models"; models are server-side entities; open-source server lacks RBAC and vault. — [clearml-server](https://github.com/clearml/clearml-server), [ClearML pricing](https://clear.ml/pricing/)
- Sacred has no registry; Omniboard is a dashboard only. — [IDSIA/sacred](https://github.com/IDSIA/sacred), [omniboard](https://github.com/vivekratnavel/omniboard)
- DagsHub pairs DVC-versioned data/models with MLflow-compatible tracking. — [dagshub.com](https://dagshub.com/)

### Inferences
- A git-tag registry (GTO pattern) plus content-addressed blobs (Trackio pattern) covers forestry's "registry in git or a flat file" requirement with two small, already-proven designs; MLflow's registry can be local but is a SQLAlchemy schema subject to `mlflow db upgrade` migrations.

### Gaps
- Comet's and W&B's registry data models were not read from primary docs (W&B docs gated; Comet registry page not fetched).

## What are the smallest viable self-hosted footprints and resource costs?

### Takeaway
Zero-process: MLflow SQLite, Trackio, DVCLive, TensorBoard files, Lightning CSV, Sacred FileStorage, mltraq. One process on demand: `mlflow ui`, `trackio show`, `aim up`, `tensorboard`. One container: W&B Server (`wandb/local`, bundles MySQL and MinIO, needs a license key). Multi-container: ClearML (Elasticsearch, MongoDB, Redis plus four services, 8 GB RAM minimum, 16 GB recommended), SwanLab (gateway, MinIO, Traefik and backing services; 2 cores, 4 GB, 20 GB), Comet on-prem (requirements not retrievable).

### Cited Findings
- MLflow: `mlflow server --backend-store-uri sqlite:///mlflow.db` is the whole deployment. — [Troubleshooting](https://mlflow.org/docs/latest/self-hosting/troubleshooting/). Independent estimate for a Postgres plus object-store deployment: "well under $100/month". — [Spheron 2026](https://www.spheron.network/blog/weights-biases-pricing-vs-self-hosted-mlflow-2026/)
- Trackio: `trackio show` launches the dashboard; optional self-hosted Trackio server via `server_url`. — [API](https://huggingface.co/docs/trackio/main/api), [Index](https://huggingface.co/docs/trackio/index)
- W&B Server: `wandb server start` or `docker run ... wandb/local` with bundled MySQL and MinIO; free license from deploy.wandb.ai; production needs external MySQL 8, S3, Redis and a paid license. — [wandb/server](https://github.com/wandb/server)
- ClearML: minimum 8 GB RAM, 16 GB recommended; Elasticsearch, MongoDB, Redis, apiserver, webserver, fileserver, agent-services, async_delete; `vm.max_map_count=524288`. — [ClearML Server Linux/macOS](http://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_linux_mac/). Server license SSPL-1.0. — [clearml-server](https://github.com/clearml/clearml-server)
- SwanLab: CPU >= 2 cores, RAM >= 4 GB, disk >= 20 GB; gateway 8000, MinIO 9000; free community license. — [SwanLab Docker deploy](https://docs.swanlab.cn/en/self_host/docker/deploy.html)
- Aim: `aim up` local server; reindex thread on startup; reports of near one-minute startup and 1-2 minute `aim.Run` creation in slow versions. — [aimhubio/aim](https://github.com/aimhubio/aim), [Lightrun aim slowness](https://lightrun.com/answers/aimhubio-aim-aim-is-very-slow-since-3120)
- Omniboard needs MongoDB >= 4.0 plus a Node process. — [omniboard](https://github.com/vivekratnavel/omniboard)
- Opik (Comet's open-source observability) documents Kubernetes as the production path. — [Opik self-host](https://www.comet.com/docs/opik/self-host/overview)

### Inferences
- Only the zero-process group fits "no always-on server" without compromise; W&B's single container is the smallest of the server products but is license-gated and its docs are now behind CoreWeave.

### Gaps
- Comet on-prem (cometctl "All in One") requirements: both the official page (404) and a public mirror (403) failed.
- W&B `wandb/local` RAM/CPU requirements not found in a public source.

## Which have a first-class CLI and Python API that an automated agent could drive without a browser?

### Takeaway
MLflow (fluent API, `MlflowClient`, `search_runs` to pandas, `mlflow` CLI incl. `db upgrade`, `gc`), Trackio (wandb-compatible API, `trackio list/get/query --json`, dashboard doubles as an MCP server), DVC (`dvc exp show --csv/--json`, `dvc exp diff/push`), ClearML (`clearml-task` CLI, `Task` API), mltraq (pure Python, `.df()`), Aim (Python-expression query SDK) and Sacred (CLI config overrides, Python observers) are all browser-free. W&B and Comet have full Python APIs and CLIs but comparison views are server-side; SwanLab exposes `swanlab.Api` and OpenAPI.

### Cited Findings
- MLflow `mlflow.search_runs` returns a pandas DataFrame with a SQL-like filter DSL. — [Search runs](https://mlflow.org/docs/latest/ml/search/search-runs/); CLI `mlflow db upgrade`, `mlflow gc`, `mlflow server`. — [Backend store](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/), [Troubleshooting](https://mlflow.org/docs/latest/self-hosting/troubleshooting/)
- Trackio: "LLM-friendly: Designed for autonomous ML experiments with CLI commands and Python APIs"; `trackio query ... --json` (read-only SELECT/WITH/PRAGMA), `trackio list`, `trackio get`; `show(mcp_server=True)`. — [Index](https://huggingface.co/docs/trackio/index), [Storage schema](https://huggingface.co/docs/trackio/v0.32.2/storage_schema), [API](https://huggingface.co/docs/trackio/main/api)
- DVC: `dvc exp show`, `dvc exp diff`, `dvc exp push`. — [Experiments overview](https://doc.dvc.org/user-guide/experiment-management/experiments-overview); `--csv`/`--json` on `dvc exp show`. — [DVCLive](https://doc.dvc.org/dvclive)
- ClearML `clearml-task` CLI and `Task.import_offline_session`. — [ClearML Task CLI](https://clear.ml/docs/latest/docs/apps/clearml_task/), [Set offline](http://clear.ml/docs/latest/docs/guides/set_offline/)
- Aim SDK: `my_repo.query_metrics(query).iter_runs()`. — [aimhubio/aim](https://github.com/aimhubio/aim)
- mltraq: `.persist()`, `.reload()`, `.df()`; SQL-queryable store. — [mltraq.com](https://www.mltraq.com/)
- W&B CLI: `wandb offline`, `wandb online`, `wandb disabled`, `wandb sync`, `wandb server start`. — [wandb offline](https://docs.wandb.ai/models/ref/cli/wandb-offline.md), [wandb disabled](https://docs.wandb.ai/models/ref/cli/wandb-disabled.md), [wandb/server](https://github.com/wandb/server)
- Comet: `comet upload` CLI and Python SDK. — [Running offline](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/advanced/running-offline/)
- SwanLab: `swanlab.Api`, OpenAPI, `swanlab watch`. — [SwanHubX/SwanLab](https://github.com/SwanHubX/SwanLab)
- Guild AI: CLI-first ("Run unmodified training scripts"), `guild view` optional. — [guildai/guildai](https://github.com/guildai/guildai)

### Inferences
- Trackio's read-only SQL CLI with `--json` and MCP exposure is the most agent-oriented surface in the set; MLflow's pandas DataFrame return is the most mature for programmatic comparison.

### Gaps
- Did not verify W&B public API (`wandb.Api().runs(...)`) behavior against a self-hosted or offline store; W&B docs were gated.

## What are the documented weaknesses: UI slowness at many runs, schema migrations, lock-in, telemetry?

### Takeaway
Documented weaknesses cluster as: scale pain (MLflow file store without indexing and the ag-grid UI issue at ~1000 runs; Aim indexing and startup slowness; W&B sync retries and multi-hour artifact syncs), migrations (MLflow `mlflow db upgrade` with downtime warning; Trackio beta schema may require regenerated databases), lock-in events (Neptune data irreversibly deleted; W&B docs and pricing moved under CoreWeave), and telemetry (MLflow usage tracking on by default since 3.2.0; comet-ml bundles sentry-sdk). Dormancy is its own weakness: Omniboard last pushed 2023, Guild AI 2025-04 with 529 downloads/month, Aim no release since v3.29.1.

### Cited Findings
- MLflow file store "severely limits the performance, e.g., no indexing". — [Troubleshooting](https://mlflow.org/docs/latest/self-hosting/troubleshooting/); UI "Load more" 59 s at 1000 runs (closed). — [mlflow#5653](https://github.com/mlflow/mlflow/issues/5653); migrations can cause downtime, back up first. — [Backend store](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/); telemetry default-on since 3.2.0, `MLFLOW_DISABLE_TELEMETRY=true`. — [Usage tracking](https://mlflow.org/docs/latest/community/usage-tracking/)
- Aim: unindexed chunks slow queries; "very slow since 3.12.0"; 480 open issues; no release since 3.29.1. — [Storage indexing](https://aimstack.readthedocs.io/en/latest/understanding/storage_indexing.html), [Lightrun](https://lightrun.com/answers/aimhubio-aim-aim-is-very-slow-since-3120), [GitHub API aim](https://api.github.com/repos/aimhubio/aim), [Aim releases](https://github.com/aimhubio/aim/releases)
- W&B sync pain (retries on deleted runs, hours for 300 MB artifacts, 2k runs/day errors). — [W&B forum 4786](https://community.wandb.ai/t/sync-local-offline-runs-to-the-dashboard-while-deleting-old-folders/4786), [W&B forum 1145](https://community.wandb.ai/t/best-practices-for-many-quick-runs/1145). Enterprise per-seat $315-$400/month with the quirk that fewer seats raised the per-seat price. — [Spheron 2026](https://www.spheron.network/blog/weights-biases-pricing-vs-self-hosted-mlflow-2026/)
- Trackio: "Trackio is still in beta ... future releases may evolve the schema and require migrations or regenerated local databases"; older databases keyed on non-unique `run_name`, newer on `run_id`. — [Storage schema](https://huggingface.co/docs/trackio/v0.32.2/storage_schema), [gradio-app/trackio](https://github.com/gradio-app/trackio)
- Neptune: hosted data deleted 2026-03-05, "no export, no restore, and no recovery path"; client archived. — [Transition hub](https://docs.neptune.ai/transition_hub), [Support article](https://support.neptune.ai/en/articles/13925165-service-shutdown-overview), [GitHub API neptune-client](https://api.github.com/repos/neptune-ai/neptune-client)
- ClearML: open-source server excludes RBAC/LDAP, vault, multi-tenancy; SSPL license on the server; Elasticsearch footprint. — [ClearML pricing](https://clear.ml/pricing/), [clearml-server](https://github.com/clearml/clearml-server), [Server Linux/macOS](http://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_linux_mac/)
- Comet SDK depends on `sentry-sdk`. — [pypistats comet-ml](https://pypistats.org/packages/comet-ml)
- Omniboard last release 2021-12-26, last push 2023-02-01; Guild AI last push 2025-04-29, 529 downloads/month; Sacred last release 2024-11-26. — [Omniboard releases](https://github.com/vivekratnavel/omniboard/releases), [GitHub API omniboard](https://api.github.com/repos/vivekratnavel/omniboard), [GitHub API guildai](https://api.github.com/repos/guildai/guildai), [pypistats guildai](https://pypistats.org/packages/guildai), [PyPI JSON sacred](https://pypi.org/pypi/sacred/json)
- TensorBoard.dev shutdown removed the only hosted sharing path; HParams table drops hyperparameters when runs have differing hparam sets. — [tensorboard.dev](https://tensorboard.dev/), [Bits of experience](https://audiolabs-erlangen.com/fau/assistant/gaznepoglu/bits_of_experience)
- ClearML offline mode excludes `Task.create()` tasks. — [Set offline](http://clear.ml/docs/latest/docs/guides/set_offline/)

### Inferences
- For a small-scale, local-first project, the weaknesses that bite are migrations and schema churn (MLflow, Trackio) rather than UI scale; both can be mitigated by owning the schema, which is the core "build" argument.
- The W&B and Neptune ownership changes within 12 months make SaaS trackers a provenance risk for a project whose results ledger must outlive vendors.

### Gaps
- No primary W&B statement on client telemetry/error reporting was retrievable (docs gated); only the system-metrics opt-out was confirmed.
- No 2025-2026 independent benchmark of UI responsiveness across trackers at 1k-10k runs was found; evidence is issue-tracker and forum anecdotes.
