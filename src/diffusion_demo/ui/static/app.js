/**
 * Diffusion Demo - Main Application JavaScript
 */

// Global state
const state = {
    currentTab: 'dataset',
    activeJobs: new Map(), // job_id -> { interval, logOffset }
    checkpoints: [],
};

// Utility functions
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);

const formatDate = (isoString) => {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleString();
};

const truncate = (str, len = 20) => {
    if (!str) return '';
    return str.length > len ? str.slice(0, len) + '...' : str;
};

// API functions
const api = {
    async get(path) {
        const response = await fetch(`/api${path}`);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP ${response.status}`);
        }
        return response.json();
    },

    async post(path, data = {}) {
        const response = await fetch(`/api${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP ${response.status}`);
        }
        return response.json();
    },
};

// Tab switching
const initTabs = () => {
    $$('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            switchTab(tabId);
        });
    });
};

const switchTab = (tabId) => {
    // Update buttons
    $$('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });

    // Update content
    $$('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === tabId);
    });

    state.currentTab = tabId;

    // Tab-specific initialization
    if (tabId === 'train-dit' || tabId === 'sample') {
        loadCheckpoints();
    } else if (tabId === 'jobs') {
        loadJobs();
    }
};

// System info
const loadSystemInfo = async () => {
    try {
        const info = await api.get('/system');
        const html = `
            PyTorch ${info.torch.version} | 
            CUDA: ${info.torch.cuda_available ? '✓' : '✗'} |
            BF16: ${info.torch.bf16_supported ? '✓' : '✗'} |
            Dataset: ${info.dataset.cached ? '✓ Cached' : '✗ Not cached'}
        `;
        $('#system-info').innerHTML = html;
    } catch (err) {
        console.error('Failed to load system info:', err);
    }
};

// Dataset tab
const initDatasetTab = () => {
    $('#btn-ensure-dataset').addEventListener('click', async () => {
        const btn = $('#btn-ensure-dataset');
        const status = $('#dataset-status');
        
        btn.disabled = true;
        status.innerHTML = '<span class="spinner"></span> Downloading...';
        
        try {
            const result = await api.post('/dataset/ensure');
            status.textContent = result.success ? '✓ Cached' : '✗ Failed';
            if (result.success) loadSystemInfo();
        } catch (err) {
            status.textContent = `✗ Error: ${err.message}`;
        } finally {
            btn.disabled = false;
        }
    });

    $('#btn-refresh-samples').addEventListener('click', loadDatasetSamples);
};

const loadDatasetSamples = async () => {
    const container = $('#dataset-samples');
    const count = $('#sample-count').value;
    const seed = $('#sample-seed').value;
    
    container.innerHTML = '<div class="spinner"></div>';
    
    try {
        const data = await api.get(`/dataset/samples?count=${count}&seed=${seed}`);
        container.innerHTML = data.images.map(img => 
            `<img src="data:image/png;base64,${img}" alt="Sample">`
        ).join('');
    } catch (err) {
        container.innerHTML = `<div class="error-message">Error: ${err.message}</div>`;
    }
};

// VAE Training tab
const initVaeTab = () => {
    // Toggle VAE/VQVAE params
    $('#vae-model-type').addEventListener('change', (e) => {
        const isVae = e.target.value === 'vae';
        $('#vae-params').classList.toggle('hidden', !isVae);
        $('#vqvae-params').classList.toggle('hidden', isVae);
    });

    $('#btn-train-vae').addEventListener('click', startVaeTraining);
};

const startVaeTraining = async () => {
    const modelType = $('#vae-model-type').value;
    
    const config = {
        model: modelType,
        lr: parseFloat($('#vae-lr').value),
        batch_size: parseInt($('#vae-batch-size').value),
        epochs: parseInt($('#vae-epochs').value),
        data_limit: parseInt($('#vae-data-limit').value),
        use_bfloat16: $('#vae-bf16').checked,
        use_gan: $('#vae-use-gan').checked,
    };

    if (modelType === 'vae') {
        config.latent_dim = parseInt($('#vae-latent-dim').value);
        config.kld_beta = parseFloat($('#vae-kld-beta').value);
    } else {
        config.embedding_dim = parseInt($('#vqvae-embedding-dim').value);
        config.num_embeddings = parseInt($('#vqvae-num-embeddings').value);
        config.commitment_cost = parseFloat($('#vqvae-commitment-cost').value);
        config.vq_beta = parseFloat($('#vqvae-vq-beta').value);
        config.vq_hidden_dims_enc = [128, 64];
        config.vq_hidden_dims_dec = [64, 128];
    }

    try {
        const result = await api.post('/jobs/vae', config);
        showJobTracker('vae', result.job_id);
        trackJob('vae', result.job_id);
    } catch (err) {
        alert(`Failed to start training: ${err.message}`);
    }
};

// DiT Training tab
const initDitTab = () => {
    $('#btn-train-dit').addEventListener('click', startDitTraining);
};

const startDitTraining = async () => {
    const vaePath = $('#dit-vae-checkpoint').value;
    if (!vaePath) {
        alert('Please select a VAE checkpoint');
        return;
    }

    const config = {
        vae_model_path: vaePath,
        vae_model_type: $('#dit-vae-type').value,
        vae_latent_channels: parseInt($('#dit-latent-channels').value),
        vae_latent_spatial_dim: parseInt($('#dit-latent-spatial').value),
        vae_flat_latent_dim: parseInt($('#dit-latent-flat').value),
        lr: parseFloat($('#dit-lr').value),
        batch_size: parseInt($('#dit-batch-size').value),
        epochs: parseInt($('#dit-epochs').value),
        data_limit: parseInt($('#dit-data-limit').value),
        ddpm_timesteps: parseInt($('#dit-timesteps').value),
        ddpm_schedule: $('#dit-schedule').value,
        use_bfloat16: $('#dit-bf16').checked,
        dit_patch_size: parseInt($('#dit-patch-size').value),
        dit_hidden_size: parseInt($('#dit-hidden-size').value),
        dit_depth: parseInt($('#dit-depth').value),
        dit_num_heads: parseInt($('#dit-heads').value),
    };

    try {
        const result = await api.post('/jobs/dit', config);
        showJobTracker('dit', result.job_id);
        trackJob('dit', result.job_id);
    } catch (err) {
        alert(`Failed to start training: ${err.message}`);
    }
};

// Sample tab
const initSampleTab = () => {
    $('#btn-generate').addEventListener('click', startSampling);
};

const startSampling = async () => {
    const vaePath = $('#sample-vae-checkpoint').value;
    const ditPath = $('#sample-dit-checkpoint').value;

    if (!vaePath || !ditPath) {
        alert('Please select both VAE and DiT checkpoints');
        return;
    }

    const config = {
        vae_model_path: vaePath,
        dit_model_path: ditPath,
        num_samples: parseInt($('#sample-num').value),
        ddpm_timesteps: parseInt($('#sample-timesteps').value),
        use_bfloat16: $('#sample-bf16').checked,
    };

    try {
        const result = await api.post('/jobs/sample', config);
        showJobTracker('sample', result.job_id);
        trackJob('sample', result.job_id);
    } catch (err) {
        alert(`Failed to start sampling: ${err.message}`);
    }
};

// Jobs tab
const initJobsTab = () => {
    $('#btn-refresh-jobs').addEventListener('click', loadJobs);
    $('#jobs-filter').addEventListener('change', loadJobs);
};

const loadJobs = async () => {
    const container = $('#jobs-list');
    const filter = $('#jobs-filter').value;
    
    try {
        const url = filter ? `/jobs?type=${filter}` : '/jobs';
        const data = await api.get(url);
        
        if (data.jobs.length === 0) {
            container.innerHTML = '<div class="job-item">No jobs found</div>';
            return;
        }
        
        container.innerHTML = data.jobs.map(job => `
            <div class="job-item" data-job-id="${job.id}">
                <div class="job-item-header">
                    <span class="job-item-type">${job.type.toUpperCase()}</span>
                    <span class="status-badge ${job.status}">${job.status}</span>
                </div>
                <div class="job-item-time">${formatDate(job.created_at)}</div>
                <div class="job-item-id">${truncate(job.id, 16)}</div>
            </div>
        `).join('');
        
        // Add click handlers
        $$('.job-item').forEach(item => {
            item.addEventListener('click', () => showJobDetail(item.dataset.jobId));
        });
    } catch (err) {
        container.innerHTML = `<div class="error-message">Error: ${err.message}</div>`;
    }
};

const showJobDetail = async (jobId) => {
    const detail = $('#job-detail');
    const content = $('#job-detail-content');
    
    detail.classList.remove('hidden');
    content.innerHTML = '<div class="spinner"></div>';
    
    try {
        const job = await api.get(`/jobs/${jobId}`);
        const logData = await api.get(`/jobs/${jobId}/log?offset=0`);
        
        content.innerHTML = `
            <div class="job-status">
                <span>Type: <strong>${job.type.toUpperCase()}</strong></span>
                <span class="status-badge ${job.status}">${job.status}</span>
            </div>
            <div>ID: ${job.id}</div>
            <div>Created: ${formatDate(job.created_at)}</div>
            <div>Started: ${formatDate(job.started_at)}</div>
            <div>Finished: ${formatDate(job.finished_at)}</div>
            <hr>
            <div><strong>Config:</strong></div>
            <pre>${JSON.stringify(job.config, null, 2)}</pre>
            ${job.error?.message ? `
                <hr>
                <div class="error-message">
                    <strong>Error:</strong> ${job.error.message}
                    <pre>${job.error.traceback || ''}</pre>
                </div>
            ` : ''}
            <hr>
            <div><strong>Artifacts:</strong></div>
            <div>Checkpoints: ${job.artifacts.checkpoints.length}</div>
            <div>Images: ${job.artifacts.images.length}</div>
            ${job.artifacts.tensorboard_logdir ? `<div>TensorBoard: ${job.artifacts.tensorboard_logdir}</div>` : ''}
            <hr>
            <div><strong>Log (first 64KB):</strong></div>
            <div class="log-container">
                <pre class="job-log">${logData.text || '(empty)'}</pre>
            </div>
        `;
    } catch (err) {
        content.innerHTML = `<div class="error-message">Error: ${err.message}</div>`;
    }
};

// Job tracking
const showJobTracker = (type, jobId) => {
    const tracker = $(`#${type}-job-tracker`);
    tracker.classList.remove('hidden');
    $(`#${type}-job-id`).textContent = truncate(jobId, 12);
    $(`#${type}-job-status`).className = 'status-badge running';
    $(`#${type}-job-status`).textContent = 'running';
    $(`#${type}-progress`).style.width = '0%';
    $(`#${type}-job-log`).textContent = '';
};

const trackJob = (type, jobId) => {
    const stateKey = `${type}-${jobId}`;
    const jobState = {
        logOffset: 0,
        interval: null,
    };
    
    state.activeJobs.set(stateKey, jobState);
    
    // Poll for updates
    jobState.interval = setInterval(async () => {
        try {
            // Get job status
            const job = await api.get(`/jobs/${jobId}`);
            
            // Update status
            const statusEl = $(`#${type}-job-status`);
            statusEl.className = `status-badge ${job.status}`;
            statusEl.textContent = job.status;
            
            // Update progress (estimate based on epochs)
            if (job.status === 'running' && job.config.epochs) {
                const progress = $(`#${type}-progress`);
                // This is approximate - real progress would need more info
            }
            
            // Update log
            const logData = await api.get(`/jobs/${jobId}/log?offset=${jobState.logOffset}`);
            if (logData.text) {
                const logEl = $(`#${type}-job-log`);
                logEl.textContent += logData.text;
                logEl.scrollTop = logEl.scrollHeight;
                jobState.logOffset = logData.next_offset;
            }
            
            // Check if finished
            if (['succeeded', 'failed', 'canceled'].includes(job.status)) {
                clearInterval(jobState.interval);
                state.activeJobs.delete(stateKey);
                
                // Show artifacts
                showArtifacts(type, job);
                
                // If succeeded, refresh checkpoints
                if (job.status === 'succeeded') {
                    loadCheckpoints();
                }
            }
        } catch (err) {
            console.error('Job tracking error:', err);
        }
    }, 1000);
};

const showArtifacts = (type, job) => {
    const container = $(`#${type}-artifacts`);
    if (!container || !job.artifacts) return;
    
    let html = '';
    
    if (job.artifacts.checkpoints?.length > 0) {
        html += '<h4>Checkpoints</h4><div class="artifact-list">';
        job.artifacts.checkpoints.forEach(ckpt => {
            html += `<div class="artifact-item" title="${ckpt}">${truncate(ckpt, 40)}</div>`;
        });
        html += '</div>';
    }
    
    if (job.artifacts.images?.length > 0) {
        html += '<h4>Images</h4>';
        job.artifacts.images.forEach(img => {
            html += `<img src="/api/jobs/${job.id}/result" style="max-width: 100%; margin-top: 10px;">`;
        });
    }
    
    container.innerHTML = html;
};

// Load checkpoints
const loadCheckpoints = async () => {
    try {
        const data = await api.get('/checkpoints');
        state.checkpoints = data.checkpoints;
        
        const vaeSelects = ['#dit-vae-checkpoint', '#sample-vae-checkpoint'];
        const ditSelect = '#sample-dit-checkpoint';
        
        const vaeOptions = data.checkpoints
            .filter(c => c.job_type === 'vae' || c.run_name?.includes('vae'))
            .map(c => `<option value="${c.path}">${truncate(c.path, 50)}</option>`)
            .join('');
        
        const ditOptions = data.checkpoints
            .filter(c => c.job_type === 'dit' || c.run_name?.includes('dit'))
            .map(c => `<option value="${c.path}">${truncate(c.path, 50)}</option>`)
            .join('');
        
        vaeSelects.forEach(sel => {
            const el = $(sel);
            if (el) {
                el.innerHTML = '<option value="">Select a checkpoint...</option>' + vaeOptions;
            }
        });
        
        const ditEl = $(ditSelect);
        if (ditEl) {
            ditEl.innerHTML = '<option value="">Select a checkpoint...</option>' + ditOptions;
        }
    } catch (err) {
        console.error('Failed to load checkpoints:', err);
    }
};

// Initialize
const init = () => {
    initTabs();
    initDatasetTab();
    initVaeTab();
    initDitTab();
    initSampleTab();
    initJobsTab();
    
    loadSystemInfo();
    loadDatasetSamples();
};

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', init);
