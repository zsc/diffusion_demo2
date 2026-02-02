"""Job management system for tracking training/sampling jobs."""

import io
import json
import sys
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from .utils.io import ensure_dir, load_json, save_json


JOBS_DIR = Path("var/jobs")


@dataclass
class JobArtifacts:
    """Artifacts produced by a job."""
    checkpoints: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    tensorboard_logdir: Optional[str] = None
    other: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'checkpoints': self.checkpoints,
            'images': self.images,
            'tensorboard_logdir': self.tensorboard_logdir,
            'other': self.other,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'JobArtifacts':
        return cls(
            checkpoints=data.get('checkpoints', []),
            images=data.get('images', []),
            tensorboard_logdir=data.get('tensorboard_logdir'),
            other=data.get('other', {}),
        )


@dataclass
class JobError:
    """Error information for a failed job."""
    message: Optional[str] = None
    traceback: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'message': self.message,
            'traceback': self.traceback,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'JobError':
        return cls(
            message=data.get('message'),
            traceback=data.get('traceback'),
        )


@dataclass
class Job:
    """Represents a training/sampling job."""
    id: str
    type: str  # vae, dit, sample, dataset
    status: str  # queued, running, succeeded, failed, canceled
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    config: Dict = field(default_factory=dict)
    artifacts: JobArtifacts = field(default_factory=JobArtifacts)
    error: JobError = field(default_factory=JobError)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'status': self.status,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'config': self.config,
            'artifacts': self.artifacts.to_dict(),
            'error': self.error.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Job':
        return cls(
            id=data['id'],
            type=data['type'],
            status=data['status'],
            created_at=data['created_at'],
            started_at=data.get('started_at'),
            finished_at=data.get('finished_at'),
            config=data.get('config', {}),
            artifacts=JobArtifacts.from_dict(data.get('artifacts', {})),
            error=JobError.from_dict(data.get('error', {})),
        )
    
    def get_job_dir(self) -> Path:
        """Get the directory for this job."""
        return JOBS_DIR / self.id
    
    def get_meta_path(self) -> Path:
        """Get path to meta.json."""
        return self.get_job_dir() / "meta.json"
    
    def get_log_path(self) -> Path:
        """Get path to log.txt."""
        return self.get_job_dir() / "log.txt"
    
    def save(self) -> None:
        """Save job metadata to disk."""
        save_json(self.get_meta_path(), self.to_dict())


class JobRunner(threading.Thread):
    """Thread that runs a job and captures output."""
    
    def __init__(self, job: Job, func: Callable, *args, **kwargs):
        super().__init__(daemon=True)
        self.job = job
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.log_file = None
        self.original_stdout = None
        self.original_stderr = None
    
    def run(self) -> None:
        """Run the job function."""
        job = self.job
        job_dir = job.get_job_dir()
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Open log file
        log_path = job.get_log_path()
        self.log_file = open(log_path, 'w', buffering=1)
        
        # Redirect stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        class TeeOutput:
            def __init__(self, stdout, stderr, file):
                self.stdout = stdout
                self.stderr = stderr
                self.file = file
            
            def write(self, data):
                self.file.write(data)
                self.file.flush()
                self.stdout.write(data)
                self.stdout.flush()
            
            def flush(self):
                self.file.flush()
                self.stdout.flush()
        
        tee = TeeOutput(self.original_stdout, self.original_stderr, self.log_file)
        sys.stdout = tee
        sys.stderr = tee
        
        try:
            # Update status
            job.status = 'running'
            job.started_at = datetime.utcnow().isoformat() + 'Z'
            job.save()
            
            # Run the function
            result = self.func(*self.args, **self.kwargs)
            
            # Update with result artifacts
            if isinstance(result, dict):
                if 'model_path' in result:
                    job.artifacts.checkpoints.append(result['model_path'])
                if 'discriminator_path' in result and result['discriminator_path']:
                    job.artifacts.checkpoints.append(result['discriminator_path'])
                if 'checkpoint_dir' in result:
                    job.artifacts.other['checkpoint_dir'] = result['checkpoint_dir']
                if 'run_name' in result:
                    job.artifacts.other['run_name'] = result['run_name']
                if 'artifacts' in result:
                    sub_artifacts = result['artifacts']
                    if isinstance(sub_artifacts, dict):
                        job.artifacts.images.extend(sub_artifacts.get('images', []))
                        if sub_artifacts.get('tensorboard_logdir'):
                            job.artifacts.tensorboard_logdir = sub_artifacts['tensorboard_logdir']
            
            job.status = 'succeeded'
            
        except Exception as e:
            job.status = 'failed'
            job.error.message = str(e)
            job.error.traceback = traceback.format_exc()
            print(f"Job failed: {e}")
            print(traceback.format_exc())
        
        finally:
            job.finished_at = datetime.utcnow().isoformat() + 'Z'
            job.save()
            
            # Restore stdout/stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            
            if self.log_file:
                self.log_file.close()


class JobManager:
    """Manages job creation, execution, and persistence."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._jobs: Dict[str, Job] = {}
        self._runners: Dict[str, JobRunner] = {}
        self._lock = threading.Lock()
        
        # Ensure jobs directory exists
        ensure_dir(JOBS_DIR)
        
        # Load existing jobs
        self._load_existing_jobs()
    
    def _load_existing_jobs(self) -> None:
        """Load existing jobs from disk."""
        if not JOBS_DIR.exists():
            return
        
        for job_dir in JOBS_DIR.iterdir():
            if job_dir.is_dir():
                meta_path = job_dir / "meta.json"
                if meta_path.exists():
                    try:
                        data = load_json(meta_path)
                        job = Job.from_dict(data)
                        self._jobs[job.id] = job
                    except Exception as e:
                        print(f"Failed to load job {job_dir.name}: {e}")
    
    def create_job(self, job_type: str, config: dict) -> Job:
        """Create a new job.
        
        Args:
            job_type: Type of job (vae, dit, sample, dataset)
            config: Job configuration
        
        Returns:
            The created Job
        """
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            type=job_type,
            status='queued',
            created_at=datetime.utcnow().isoformat() + 'Z',
            config=config,
        )
        
        with self._lock:
            self._jobs[job_id] = job
        
        # Save immediately
        job.save()
        
        return job
    
    def start_job(self, job: Job, func: Callable, *args, **kwargs) -> None:
        """Start a job in a background thread.
        
        Args:
            job: The job to start
            func: Function to run
            *args, **kwargs: Arguments for the function
        """
        runner = JobRunner(job, func, *args, **kwargs)
        
        with self._lock:
            self._runners[job.id] = runner
        
        runner.start()
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)
    
    def list_jobs(
        self,
        job_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Job]:
        """List jobs, sorted by creation time (newest first).
        
        Args:
            job_type: Filter by type
            limit: Maximum number of jobs to return
            offset: Offset for pagination
        
        Returns:
            List of Job objects
        """
        jobs = list(self._jobs.values())
        
        # Filter by type
        if job_type:
            jobs = [j for j in jobs if j.type == job_type]
        
        # Sort by created_at (descending)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        # Apply offset and limit
        return jobs[offset:offset + limit]
    
    def get_log(self, job_id: str, offset: int = 0) -> dict:
        """Get log content for a job.
        
        Args:
            job_id: Job ID
            offset: Byte offset to start reading from
        
        Returns:
            Dict with 'text', 'next_offset', 'eof'
        """
        job = self.get_job(job_id)
        if not job:
            return {'text': '', 'next_offset': 0, 'eof': True, 'error': 'Job not found'}
        
        log_path = job.get_log_path()
        if not log_path.exists():
            return {'text': '', 'next_offset': 0, 'eof': True}
        
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(offset)
                text = f.read(65536)  # Read up to 64KB
                next_offset = f.tell()
                eof = len(text) < 65536
                
                return {
                    'text': text,
                    'next_offset': next_offset,
                    'eof': eof,
                }
        except Exception as e:
            return {'text': '', 'next_offset': offset, 'eof': True, 'error': str(e)}
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.
        
        Note: This only marks the job as canceled. The actual thread
        may continue running.
        
        Returns:
            True if job was canceled, False otherwise
        """
        job = self.get_job(job_id)
        if not job or job.status != 'running':
            return False
        
        job.status = 'canceled'
        job.finished_at = datetime.utcnow().isoformat() + 'Z'
        job.save()
        
        return True


# Global job manager instance
job_manager = JobManager()
