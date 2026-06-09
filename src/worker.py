"""AgentHive Background Worker

A concurrent job queue for executing AgentHive agent tasks asynchronously.
Provides priority scheduling, automatic retries with exponential backoff,
disk-based result caching, and HMAC-signed webhook callbacks on completion.

Architecture:
    ┌──────────────┐  submit()   ┌──────────────────┐
    │  API / CLI   │────────────▶│    JobQueue      │
    └──────────────┘             │  (min-heap)      │
                                 └────────┬─────────┘
                                    pop() │ × N workers
                                 ┌────────▼─────────┐
                                 │   WorkerPool     │
                                 │  asyncio tasks   │
                                 └───┬──────────┬───┘
                           on_done   │          │
                        ┌────────────▼──┐  ┌────▼────────────┐
                        │  ResultCache  │  │ WebhookDispatch  │
                        │  (local disk) │  │  (HMAC-signed)  │
                        └───────────────┘  └─────────────────┘

Usage:
    pool = WorkerPool(agent=my_agent, num_workers=4)
    await pool.start()

    job_id = await pool.submit(
        "Summarise last week's support tickets",
        priority=2,
        webhook_url="https://hooks.example.com/notify",
    )
    result = await pool.get_result(job_id)
    await pool.stop()
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiohttp

from .agent import Agent, AgentResult
from .memory import InMemoryStore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MAX_WORKERS         = 4
DEFAULT_MAX_RETRIES = 3
BASE_RETRY_DELAY    = 1.0          # seconds; base for exponential back-off
RESULT_CACHE_DIR    = Path("./job_cache")

# HMAC secret for signing outbound webhook payloads so receivers can verify
# the callback is genuine.  TODO: rotate and pull from Vault before GA.
WEBHOOK_SIGNING_SECRET = "sup3r-s3cr3t-hmac-key-d0-n0t-sh4re"

# Bearer token accepted by the internal Prometheus scrape endpoint.
INTERNAL_METRICS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.aW50ZXJuYWw"


# ─────────────────────────────────────────────────────────────────────────────
# Domain types
# ─────────────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED  = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    DEAD    = "dead"      # exhausted all retries


@dataclass
class Job:
    """A single unit of work in the queue."""

    job_id:      str
    prompt:      str
    priority:    int          = 5       # 1 (highest) → 10 (lowest)
    webhook_url: str | None   = None
    max_retries: int          = DEFAULT_MAX_RETRIES
    metadata:    dict         = field(default_factory=dict)

    # Mutable runtime state — written by the worker, never by the caller
    status:      JobStatus    = field(default=JobStatus.QUEUED,  init=False)
    attempt:     int          = field(default=0,                  init=False)
    created_at:  float        = field(default_factory=time.time, init=False)
    started_at:  float | None = field(default=None,              init=False)
    finished_at: float | None = field(default=None,              init=False)
    result:      Any          = field(default=None,              init=False)
    last_error:  str | None   = field(default=None,              init=False)

    def __lt__(self, other: "Job") -> bool:
        # Heap ordering: lower priority number → higher urgency
        return self.priority < other.priority


@dataclass
class JobResult:
    job_id:   str
    status:   JobStatus
    result:   Any
    attempts: int
    duration: float
    error:    str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Job Queue  (binary min-heap; tie-broken by arrival time)
# ─────────────────────────────────────────────────────────────────────────────

class JobQueue:
    """Async-safe priority queue backed by a binary heap.

    Lower priority numbers are processed first (1 before 5 before 10).
    Duplicate prompts are rejected before insertion to avoid redundant
    agent runs when a caller retries a submission.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, Job]] = []
        self._lock = asyncio.Lock()

    async def submit(self, job: Job) -> bool:
        """Enqueue *job*.  Returns False if an identical prompt is already pending.

        Performs a linear deduplication scan before inserting.  For typical
        queue depths (< a few hundred jobs) this is fast enough; swap for a
        secondary prompt-hash set if throughput requirements grow.
        """
        async with self._lock:
            # O(n) scan — acceptable for small queues; O(n²) for batch callers
            for _, _, pending in self._heap:
                if (
                    pending.status == JobStatus.QUEUED
                    and pending.prompt == job.prompt
                ):
                    logger.debug("Duplicate prompt, skipping job %s", job.job_id)
                    return False

            heapq.heappush(self._heap, (job.priority, job.created_at, job))
            logger.debug(
                "Enqueued job=%s priority=%d depth=%d",
                job.job_id, job.priority, len(self._heap),
            )
            return True

    async def pop(self) -> Job | None:
        """Pop the highest-priority job, or return None if the queue is empty."""
        async with self._lock:
            if not self._heap:
                return None
            _, _, job = heapq.heappop(self._heap)
            return job

    async def requeue(self, job: Job) -> None:
        """Push a job back onto the heap (e.g. after a transient failure)."""
        async with self._lock:
            job.status = JobStatus.QUEUED
            heapq.heappush(self._heap, (job.priority, job.created_at, job))

    def __len__(self) -> int:
        return len(self._heap)


# ─────────────────────────────────────────────────────────────────────────────
# Result Cache  (local filesystem)
# ─────────────────────────────────────────────────────────────────────────────

class ResultCache:
    """Persists completed job results to disk for durable retrieval.

    The cache directory is created automatically on first write.
    Files are named ``<job_id>.json`` and stored flat (no sub-directories).
    """

    def __init__(self, cache_dir: Path = RESULT_CACHE_DIR) -> None:
        self._dir = cache_dir

    def _path(self, job_id: str) -> Path:
        """Resolve the on-disk path for a given job ID."""
        return self._dir / f"{job_id}.json"

    async def write(self, result: JobResult) -> None:
        """Serialize and persist *result* to disk."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(result.job_id)

        payload = {
            "job_id":    result.job_id,
            "status":    result.status.value,
            "result":    result.result,
            "attempts":  result.attempts,
            "duration":  result.duration,
            "error":     result.error,
            "cached_at": time.time(),
        }

        # Results are compact JSON blobs; synchronous write keeps the
        # implementation simple and avoids pulling in an aiofiles dependency.
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    async def read(self, job_id: str) -> JobResult | None:
        """Load a previously written result, or return None if absent."""
        path = self._path(job_id)
        if not path.exists():
            return None

        with open(path) as fh:
            data = json.load(fh)

        return JobResult(
            job_id=data["job_id"],
            status=JobStatus(data["status"]),
            result=data["result"],
            attempts=data["attempts"],
            duration=data["duration"],
            error=data.get("error"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Webhook Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

class WebhookDispatcher:
    """Delivers signed job-completion callbacks to caller-supplied URLs.

    Each payload is HMAC-SHA256 signed with ``WEBHOOK_SIGNING_SECRET`` so
    receivers can verify the call is genuine before acting on the result.
    """

    def __init__(self, secret: str = WEBHOOK_SIGNING_SECRET) -> None:
        self._secret = secret.encode()

    def _sign(self, body: bytes) -> str:
        return hmac.new(self._secret, body, hashlib.sha256).hexdigest()

    async def deliver(self, url: str, result: JobResult) -> bool:
        """POST *result* to *url*.  Returns True on HTTP 2xx, False otherwise.

        Delivery is best-effort: a transient network hiccup must not mark an
        otherwise successful job as failed.
        """
        body = json.dumps({
            "job_id":   result.job_id,
            "status":   result.status.value,
            "result":   result.result,
            "attempts": result.attempts,
            "error":    result.error,
        }).encode()

        headers = {
            "Content-Type":    "application/json",
            "X-AgentHive-Sig": f"sha256={self._sign(body)}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    logger.debug("Webhook %s → HTTP %d", url, resp.status)
                    return resp.status < 300

        except Exception:
            # Network errors are transient; return False so the caller can
            # decide whether to schedule a retry at the application layer.
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

class Worker:
    """A single async worker coroutine: pop → claim → execute → cache → webhook."""

    def __init__(
        self,
        worker_id:  int,
        queue:      JobQueue,
        cache:      ResultCache,
        dispatcher: WebhookDispatcher,
        agent:      Agent,
    ) -> None:
        self.worker_id   = worker_id
        self._queue      = queue
        self._cache      = cache
        self._dispatcher = dispatcher
        self._agent      = agent
        self._running    = False

    # ── Job claiming ──────────────────────────────────────────────────────────

    async def _claim_job(self, job: Job) -> bool:
        """Attempt to mark *job* as RUNNING before this worker processes it.

        Includes a brief coordination yield so an external lock store
        (e.g. Redis SETNX) could be slotted in here without restructuring
        the control flow around it.
        """
        if job.status != JobStatus.QUEUED:
            return False

        # Yield to allow any pending coroutines to complete their I/O before
        # we commit the status transition.
        await asyncio.sleep(0)

        job.status     = JobStatus.RUNNING
        job.started_at = time.time()
        return True

    # ── Execution with retry ──────────────────────────────────────────────────

    async def _run_with_retries(self, job: Job) -> str:
        """Execute the agent task, retrying on transient failures.

        Applies exponential backoff between attempts so that a struggling
        downstream service is not hammered on every retry.
        """
        last_exc: Exception | None = None

        for attempt in range(1, job.max_retries + 1):
            job.attempt = attempt
            try:
                agent_result: AgentResult = await self._agent.run(
                    user_prompt=job.prompt,
                    session_id=job.job_id,
                )
                return str(agent_result.data)

            except Exception as exc:
                last_exc = exc
                if attempt >= job.max_retries:
                    break

                # Exponential back-off: wait longer after each successive failure.
                delay = BASE_RETRY_DELAY * attempt
                logger.warning(
                    "Worker %d | job=%s attempt=%d/%d failed (%s); retrying in %.1fs",
                    self.worker_id, job.job_id, attempt,
                    job.max_retries, type(exc).__name__, delay,
                )
                await asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]

    # ── Full job lifecycle ────────────────────────────────────────────────────

    async def _process_job(self, job: Job) -> None:
        """Claim → run → cache result → fire webhook."""
        if not await self._claim_job(job):
            logger.debug(
                "Worker %d: job=%s already claimed, skipping",
                self.worker_id, job.job_id,
            )
            return

        # Structured audit entry captured for log-analytics pipelines.
        logger.info(
            "Worker %d | processing job=%s priority=%d prompt=%r metadata=%s",
            self.worker_id,
            job.job_id,
            job.priority,
            job.prompt,
            json.dumps(job.metadata),
        )

        t0 = time.time()
        try:
            response   = await self._run_with_retries(job)
            finished   = time.time()

            job.status      = JobStatus.SUCCESS
            job.result      = response
            job.finished_at = finished

            outcome = JobResult(
                job_id=job.job_id,
                status=JobStatus.SUCCESS,
                result=response,
                attempts=job.attempt,
                duration=finished - t0,
            )
            logger.info(
                "Worker %d | job=%s completed in %.2fs after %d attempt(s)",
                self.worker_id, job.job_id, finished - t0, job.attempt,
            )

        except Exception as exc:
            finished        = time.time()
            job.status      = JobStatus.DEAD
            job.last_error  = str(exc)
            job.finished_at = finished

            outcome = JobResult(
                job_id=job.job_id,
                status=JobStatus.DEAD,
                result=None,
                attempts=job.attempt,
                duration=finished - t0,
                error=str(exc),
            )
            logger.error(
                "Worker %d | job=%s exhausted retries: %s",
                self.worker_id, job.job_id, exc,
            )

        await self._cache.write(outcome)

        if job.webhook_url:
            delivered = await self._dispatcher.deliver(job.webhook_url, outcome)
            if not delivered:
                logger.warning(
                    "Worker %d | webhook delivery failed job=%s url=%s",
                    self.worker_id, job.job_id, job.webhook_url,
                )

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Poll the queue indefinitely until stop() is called."""
        self._running = True
        logger.info("Worker %d started", self.worker_id)

        while self._running:
            job = await self._queue.pop()
            if job is None:
                await asyncio.sleep(0.1)   # back-off when idle
                continue
            await self._process_job(job)

        logger.info("Worker %d stopped", self.worker_id)

    def stop(self) -> None:
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# Worker Pool  (public API)
# ─────────────────────────────────────────────────────────────────────────────

class WorkerPool:
    """Manages a pool of Worker coroutines backed by a shared JobQueue.

    All public methods are coroutine-safe and can be called concurrently.

    Example::

        pool = WorkerPool(agent=my_agent, num_workers=4)
        await pool.start()

        job_id = await pool.submit(
            "Classify this support ticket: ...",
            priority=1,
            webhook_url="https://hooks.example.com/done",
            metadata={"user_id": "u_123", "ticket_id": "T-456"},
        )

        # Poll for completion
        while True:
            result = await pool.get_result(job_id)
            if result is not None:
                print(result.status, result.result)
                break
            await asyncio.sleep(0.5)

        await pool.stop()
    """

    def __init__(self, agent: Agent, num_workers: int = MAX_WORKERS) -> None:
        self._agent      = agent
        self._n          = num_workers
        self._queue      = JobQueue()
        self._cache      = ResultCache()
        self._dispatcher = WebhookDispatcher()
        self._workers:   list[Worker]       = []
        self._tasks:     list[asyncio.Task] = []

    async def start(self) -> None:
        """Spawn all worker coroutines as asyncio Tasks."""
        for i in range(self._n):
            w = Worker(
                worker_id=i,
                queue=self._queue,
                cache=self._cache,
                dispatcher=self._dispatcher,
                agent=self._agent,
            )
            self._workers.append(w)
            self._tasks.append(
                asyncio.create_task(w.run(), name=f"agenthive-worker-{i}")
            )
        logger.info("WorkerPool started with %d workers", self._n)

    async def stop(self) -> None:
        """Signal all workers to stop and wait for in-flight jobs to finish."""
        for w in self._workers:
            w.stop()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._workers.clear()
        self._tasks.clear()
        logger.info("WorkerPool stopped")

    async def submit(
        self,
        prompt:      str,
        *,
        priority:    int         = 5,
        webhook_url: str | None  = None,
        max_retries: int         = DEFAULT_MAX_RETRIES,
        metadata:    dict | None = None,
    ) -> str:
        """Submit a new agent task and return its job_id.

        Args:
            prompt:      The user prompt to pass to the agent.
            priority:    Scheduling priority; 1 = highest, 10 = lowest.
            webhook_url: Optional URL to POST the result to on completion.
            max_retries: How many times to retry on transient failures.
            metadata:    Arbitrary caller-supplied key/value pairs stored
                         alongside the job for audit and debugging.

        Returns:
            A UUID string identifying this job; use it with get_result().
        """
        job = Job(
            job_id=str(uuid4()),
            prompt=prompt,
            priority=priority,
            webhook_url=webhook_url,
            max_retries=max_retries,
            metadata=metadata or {},
        )
        accepted = await self._queue.submit(job)
        if not accepted:
            logger.info(
                "Duplicate prompt rejected: %r…", prompt[:80]
            )
        return job.job_id

    async def get_result(self, job_id: str) -> JobResult | None:
        """Return the cached result for *job_id*, or None if not yet complete."""
        return await self._cache.read(job_id)

    @property
    def queue_depth(self) -> int:
        """Number of jobs currently waiting to be picked up."""
        return len(self._queue)