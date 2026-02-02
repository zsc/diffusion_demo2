"""JSON API routes."""

import os
from pathlib import Path

from flask import Blueprint, jsonify, request, send_from_directory

from .core.data_loader import (
    CACHE_FILE,
    ensure_cache,
    get_dataloader,
    tensor_to_pil,
)
from .core.training.train_dit import sample_images
from .core.training.train_vae import train_vae_model
from .core.training.train_dit import train_dit_model
from .jobs import job_manager
from .utils.io import to_base64_png
from .utils.torch_env import get_env_config, get_torch_info

api_bp = Blueprint('api', __name__, url_prefix='/api')


@api_bp.route('/system')
def get_system_info():
    """Get system information."""
    torch_info = get_torch_info()
    
    # Check if dataset is cached
    cached = CACHE_FILE.exists()
    
    return jsonify({
        'torch': torch_info,
        'paths': {
            'cache': 'cache',
            'checkpoints': 'checkpoints',
            'runs': 'runs',
            'results': 'results',
        },
        'dataset': {
            'image_size': 48,
            'cached': cached,
            'cached_file': str(CACHE_FILE) if cached else None,
        },
    })


@api_bp.route('/dataset/ensure', methods=['POST'])
def ensure_dataset():
    """Ensure dataset is downloaded and cached."""
    try:
        ensure_cache()
        return jsonify({'success': True, 'cached_file': str(CACHE_FILE)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/dataset/samples')
def get_dataset_samples():
    """Get random samples from the dataset."""
    count = request.args.get('count', 16, type=int)
    seed = request.args.get('seed', 0, type=int)
    
    try:
        from .utils.torch_env import setup_seed
        setup_seed(seed)
        
        # Load dataset
        from .core.data_loader import CelebASpritesDataset
        dataset = CelebASpritesDataset()
        
        # Get random samples
        import random
        indices = random.sample(range(len(dataset)), min(count, len(dataset)))
        
        images = []
        for idx in indices:
            img = dataset.get_pil_image(idx)
            images.append(to_base64_png(img))
        
        return jsonify({
            'images': images,
            'image_size': 48,
            'count': len(images),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/jobs')
def list_jobs():
    """List jobs."""
    job_type = request.args.get('type')
    limit = request.args.get('limit', 50, type=int)
    
    jobs = job_manager.list_jobs(job_type=job_type, limit=limit)
    
    return jsonify({
        'jobs': [job.to_dict() for job in jobs],
        'total': len(jobs),
    })


@api_bp.route('/jobs/<job_id>')
def get_job(job_id):
    """Get a single job."""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job.to_dict())


@api_bp.route('/jobs/<job_id>/log')
def get_job_log(job_id):
    """Get job log."""
    offset = request.args.get('offset', 0, type=int)
    
    result = job_manager.get_log(job_id, offset)
    return jsonify(result)


@api_bp.route('/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Cancel a job."""
    success = job_manager.cancel_job(job_id)
    if not success:
        return jsonify({'error': 'Job not found or not running'}), 404
    
    return jsonify({'success': True})


@api_bp.route('/jobs/<job_id>/result')
def get_job_result(job_id):
    """Get job result image as base64."""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    # Get the latest image artifact
    if job.artifacts.images:
        latest_image = job.artifacts.images[-1]
        from PIL import Image
        img = Image.open(latest_image)
        return jsonify({
            'image': to_base64_png(img),
            'path': latest_image,
        })
    
    return jsonify({'error': 'No result image available'}), 404


@api_bp.route('/jobs/vae', methods=['POST'])
def create_vae_job():
    """Create a VAE/VQVAE training job."""
    data = request.get_json()
    
    model_type = data.get('model', 'vae')
    lr = data.get('lr', 0.0002)
    batch_size = data.get('batch_size', 128)
    epochs = data.get('epochs', 5)
    data_limit = data.get('data_limit', 2048)
    use_bfloat16 = data.get('use_bfloat16', True)
    use_gan = data.get('use_gan', False)
    
    # VAE-specific params
    latent_dim = data.get('latent_dim', 256)
    kld_beta = data.get('kld_beta', 1.0)
    
    # VQVAE-specific params
    embedding_dim = data.get('embedding_dim', 64)
    num_embeddings = data.get('num_embeddings', 512)
    commitment_cost = data.get('commitment_cost', 0.25)
    vq_beta = data.get('vq_beta', 1.0)
    vq_hidden_dims_enc = data.get('vq_hidden_dims_enc', [128, 64])
    vq_hidden_dims_dec = data.get('vq_hidden_dims_dec', [64, 128])
    
    config = {
        'model': model_type,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'data_limit': data_limit,
        'use_bfloat16': use_bfloat16,
        'use_gan': use_gan,
        'latent_dim': latent_dim,
        'kld_beta': kld_beta,
        'embedding_dim': embedding_dim,
        'num_embeddings': num_embeddings,
        'commitment_cost': commitment_cost,
        'vq_beta': vq_beta,
        'vq_hidden_dims_enc': vq_hidden_dims_enc,
        'vq_hidden_dims_dec': vq_hidden_dims_dec,
    }
    
    job = job_manager.create_job('vae', config)
    
    # Start training in background
    job_manager.start_job(
        job,
        train_vae_model,
        model_type=model_type,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        data_limit=data_limit,
        use_bfloat16=use_bfloat16,
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        kld_beta=kld_beta,
        vq_beta=vq_beta,
        vq_hidden_dims_enc=vq_hidden_dims_enc,
        vq_hidden_dims_dec=vq_hidden_dims_dec,
        use_gan=use_gan,
    )
    
    return jsonify({'job_id': job.id})


@api_bp.route('/jobs/dit', methods=['POST'])
def create_dit_job():
    """Create a DiT training job."""
    data = request.get_json()
    
    vae_model_path = data.get('vae_model_path')
    if not vae_model_path:
        return jsonify({'error': 'vae_model_path is required'}), 400
    
    if not os.path.exists(vae_model_path):
        return jsonify({'error': f'VAE model not found: {vae_model_path}'}), 400
    
    discriminator_model_path = data.get('discriminator_model_path')
    use_gan = data.get('use_gan', False)
    
    if use_gan and discriminator_model_path and not os.path.exists(discriminator_model_path):
        return jsonify({'error': f'Discriminator model not found: {discriminator_model_path}'}), 400
    
    gan_loss_weight = data.get('gan_loss_weight', 0.01)
    lr_d = data.get('lr_d', 0.0001)
    ddpm_schedule = data.get('ddpm_schedule', 'linear')
    ddpm_timesteps = data.get('ddpm_timesteps', 1000)
    lr = data.get('lr', 0.00001)
    batch_size = data.get('batch_size', 64)
    epochs = data.get('epochs', 5)
    data_limit = data.get('data_limit', 2048)
    use_bfloat16 = data.get('use_bfloat16', True)
    dit_patch_size = data.get('dit_patch_size', 2)
    dit_hidden_size = data.get('dit_hidden_size', 384)
    dit_depth = data.get('dit_depth', 6)
    dit_num_heads = data.get('dit_num_heads', 6)
    vae_latent_is_flat = data.get('vae_latent_is_flat', True)
    vae_flat_latent_dim = data.get('vae_flat_latent_dim', 256)
    vae_latent_channels = data.get('vae_latent_channels', 16)
    vae_latent_spatial_dim = data.get('vae_latent_spatial_dim', 4)
    vae_latent_clamp_range = data.get('vae_latent_clamp_range', [-2.0, 2.0])
    vae_model_type = data.get('vae_model_type', 'vae')
    
    config = {
        'vae_model_path': vae_model_path,
        'discriminator_model_path': discriminator_model_path,
        'use_gan': use_gan,
        'gan_loss_weight': gan_loss_weight,
        'lr_d': lr_d,
        'ddpm_schedule': ddpm_schedule,
        'ddpm_timesteps': ddpm_timesteps,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'data_limit': data_limit,
        'use_bfloat16': use_bfloat16,
        'dit_patch_size': dit_patch_size,
        'dit_hidden_size': dit_hidden_size,
        'dit_depth': dit_depth,
        'dit_num_heads': dit_num_heads,
        'vae_latent_is_flat': vae_latent_is_flat,
        'vae_flat_latent_dim': vae_flat_latent_dim,
        'vae_latent_channels': vae_latent_channels,
        'vae_latent_spatial_dim': vae_latent_spatial_dim,
        'vae_latent_clamp_range': vae_latent_clamp_range,
        'vae_model_type': vae_model_type,
    }
    
    job = job_manager.create_job('dit', config)
    
    # Start training in background
    job_manager.start_job(
        job,
        train_dit_model,
        vae_model_path=vae_model_path,
        discriminator_model_path=discriminator_model_path,
        use_gan=use_gan,
        gan_loss_weight=gan_loss_weight,
        lr_d=lr_d,
        ddpm_schedule=ddpm_schedule,
        ddpm_timesteps=ddpm_timesteps,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        data_limit=data_limit,
        use_bfloat16=use_bfloat16,
        dit_patch_size=dit_patch_size,
        dit_hidden_size=dit_hidden_size,
        dit_depth=dit_depth,
        dit_num_heads=dit_num_heads,
        vae_latent_is_flat=vae_latent_is_flat,
        vae_flat_latent_dim=vae_flat_latent_dim,
        vae_latent_channels=vae_latent_channels,
        vae_latent_spatial_dim=vae_latent_spatial_dim,
        vae_latent_clamp_range=vae_latent_clamp_range,
        vae_model_type=vae_model_type,
    )
    
    return jsonify({'job_id': job.id})


@api_bp.route('/jobs/sample', methods=['POST'])
def create_sample_job():
    """Create a sampling job."""
    data = request.get_json()
    
    vae_model_path = data.get('vae_model_path')
    dit_model_path = data.get('dit_model_path')
    
    if not vae_model_path or not os.path.exists(vae_model_path):
        return jsonify({'error': f'VAE model not found: {vae_model_path}'}), 400
    
    if not dit_model_path or not os.path.exists(dit_model_path):
        return jsonify({'error': f'DiT model not found: {dit_model_path}'}), 400
    
    num_samples = data.get('num_samples', 16)
    ddpm_timesteps = data.get('ddpm_timesteps', 1000)
    use_bfloat16 = data.get('use_bfloat16', True)
    dit_patch_size = data.get('dit_patch_size', 2)
    dit_hidden_size = data.get('dit_hidden_size', 384)
    dit_depth = data.get('dit_depth', 6)
    dit_num_heads = data.get('dit_num_heads', 6)
    vae_latent_channels = data.get('vae_latent_channels', 16)
    vae_latent_spatial_dim = data.get('vae_latent_spatial_dim', 4)
    vae_flat_latent_dim = data.get('vae_flat_latent_dim', 256)
    vq_embedding_dim_for_dit = data.get('vq_embedding_dim_for_dit', 64)
    vae_model_type = data.get('vae_model_type', 'vae')
    
    config = {
        'vae_model_path': vae_model_path,
        'dit_model_path': dit_model_path,
        'num_samples': num_samples,
        'ddpm_timesteps': ddpm_timesteps,
        'use_bfloat16': use_bfloat16,
        'dit_patch_size': dit_patch_size,
        'dit_hidden_size': dit_hidden_size,
        'dit_depth': dit_depth,
        'dit_num_heads': dit_num_heads,
        'vae_latent_channels': vae_latent_channels,
        'vae_latent_spatial_dim': vae_latent_spatial_dim,
        'vae_flat_latent_dim': vae_flat_latent_dim,
        'vq_embedding_dim_for_dit': vq_embedding_dim_for_dit,
        'vae_model_type': vae_model_type,
    }
    
    job = job_manager.create_job('sample', config)
    
    def do_sampling():
        """Run sampling and save result."""
        from pathlib import Path
        from PIL import Image
        
        samples = sample_images(
            vae_model_path=vae_model_path,
            dit_model_path=dit_model_path,
            num_samples=num_samples,
            ddpm_timesteps=ddpm_timesteps,
            use_bfloat16=use_bfloat16,
            dit_patch_size=dit_patch_size,
            dit_hidden_size=dit_hidden_size,
            dit_depth=dit_depth,
            dit_num_heads=dit_num_heads,
            vae_latent_channels=vae_latent_channels,
            vae_latent_spatial_dim=vae_latent_spatial_dim,
            vae_flat_latent_dim=vae_flat_latent_dim,
            vq_embedding_dim_for_dit=vq_embedding_dim_for_dit,
            vae_model_type=vae_model_type,
        )
        
        # Save samples
        sample_dir = Path('results/generated')
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        img_path = sample_dir / f"{job.id}.png"
        img = tensor_to_pil(samples)
        img.save(img_path)
        
        return {
            'model_path': None,
            'artifacts': {
                'images': [str(img_path)],
            }
        }
    
    # Start sampling in background
    job_manager.start_job(job, do_sampling)
    
    return jsonify({'job_id': job.id})


@api_bp.route('/checkpoints')
def list_checkpoints():
    """List available checkpoints from previous jobs."""
    checkpoints = []
    
    # Get from job artifacts
    jobs = job_manager.list_jobs(limit=100)
    for job in jobs:
        for ckpt in job.artifacts.checkpoints:
            if os.path.exists(ckpt):
                checkpoints.append({
                    'path': ckpt,
                    'job_id': job.id,
                    'job_type': job.type,
                    'created_at': job.created_at,
                })
    
    # Also scan checkpoints directory
    ckpt_dir = Path('checkpoints')
    if ckpt_dir.exists():
        for run_dir in ckpt_dir.iterdir():
            if run_dir.is_dir():
                final_model = run_dir / 'final_model.pth'
                if final_model.exists():
                    # Check if already in list
                    if not any(c['path'] == str(final_model) for c in checkpoints):
                        checkpoints.append({
                            'path': str(final_model),
                            'job_id': None,
                            'job_type': 'unknown',
                            'created_at': None,
                            'run_name': run_dir.name,
                        })
    
    return jsonify({'checkpoints': checkpoints})
