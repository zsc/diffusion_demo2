"""Command-line training runner with metrics tracking and visualization."""

import base64
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from diffusion_demo.core.training.train_vae import train_vae_model
from diffusion_demo.utils.torch_env import get_torch_info


class MetricsTracker:
    """Track training metrics in real-time."""
    
    def __init__(self, log_dir: str = "runs/cmdline"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train_start_time': None,
            'train_end_time': None,
            'epochs': [],
            'batches': [],
            'gpu_usage': [],
            'system_info': get_torch_info(),
        }
        
        self.current_epoch = 0
        self.epoch_start_time = None
        
    def start_training(self):
        """Mark training start."""
        self.metrics['train_start_time'] = datetime.now().isoformat()
        print(f"[{self.metrics['train_start_time']}] Training started")
        
    def start_epoch(self, epoch: int):
        """Mark epoch start."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}")
        print(f"{'='*60}")
        
    def log_batch(self, **kwargs):
        """Log batch metrics."""
        entry = {
            'epoch': self.current_epoch,
            'timestamp': time.time(),
        }
        entry.update(kwargs)
        
        # Track GPU usage
        if torch.cuda.is_available():
            entry['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            entry['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
            # GPU utilization requires pynvml, skip if not available
            try:
                entry['gpu_utilization_percent'] = torch.cuda.utilization()
            except:
                entry['gpu_utilization_percent'] = 0
        
        self.metrics['batches'].append(entry)
        
    def end_epoch(self, epoch_loss: float, **kwargs):
        """Mark epoch end."""
        epoch_time = time.time() - self.epoch_start_time
        
        entry = {
            'epoch': self.current_epoch,
            'avg_loss': epoch_loss,
            'epoch_time_sec': epoch_time,
            'timestamp': datetime.now().isoformat(),
        }
        entry.update(kwargs)
        
        self.metrics['epochs'].append(entry)
        
        print(f"Epoch {self.current_epoch + 1} finished:")
        print(f"  Avg Loss: {epoch_loss:.6f}")
        print(f"  Time: {epoch_time:.2f}s")
        for k, v in kwargs.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            elif isinstance(v, list):
                print(f"  {k}: {len(v)} items")
            else:
                print(f"  {k}: {v}")
            
    def end_training(self):
        """Mark training end."""
        self.metrics['train_end_time'] = datetime.now().isoformat()
        
        # Calculate total time
        start = datetime.fromisoformat(self.metrics['train_start_time'])
        end = datetime.fromisoformat(self.metrics['train_end_time'])
        total_time = (end - start).total_seconds()
        self.metrics['total_time_sec'] = total_time
        
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time:.2f}s")
        print(f"{'='*60}")
        
    def save(self, filename: str = "metrics.json"):
        """Save metrics to JSON."""
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")
        return filepath


def train_vae_cmdline(
    model_type: str = "vae", epochs: int = 10, batch_size: int = 128,
    data_limit: int = 2048, lr: float = 0.0002, use_bfloat16: bool = False,
    latent_dim: int = 256, device: str = None, precision: str = None,
    seed: int = 0, kld_beta: float = 1.0, output_dir: str = None,
    **training_options,
):
    """Use the same trainer, holdout split and checkpoint format as the API."""
    started = datetime.now().isoformat()
    result = train_vae_model(
        model_type=model_type, epochs=epochs, batch_size=batch_size, data_limit=data_limit,
        lr=lr, use_bfloat16=use_bfloat16, latent_dim=latent_dim, device=device,
        precision=precision, seed=seed, kld_beta=kld_beta, output_dir=output_dir,
        **training_options,
    )
    summary = json.loads(Path(result['metrics_path']).read_text())
    metrics = dict(train_start_time=started, train_end_time=datetime.now().isoformat(),
                   total_time_sec=summary['total_seconds'], system_info=get_torch_info(),
                   config=summary['config'], validation_images=summary['validation_images'],
                   epochs=[], batches=[])
    for record, image_path in zip(summary['epochs'], result['artifacts']['images']):
        metrics['epochs'].append(dict(
            **{k: v for k, v in record.items() if k != 'epoch'},
            epoch=record['epoch'] - 1, avg_loss=record['loss'],
            epoch_time_sec=record['train_seconds'],
            recon_images=[base64.b64encode(Path(image_path).read_bytes()).decode('ascii')],
            generated_images=[],
        ))
    metrics['total_batches'] = sum(e['optimizer_updates'] + e['skipped_updates'] for e in metrics['epochs'])
    root = Path(result['metrics_path']).parent
    (root / 'report_metrics.json').write_text(json.dumps(metrics, indent=2))
    result.update(html_path=str(generate_html_report(metrics, root / 'report.html')), metrics=metrics)
    return result


def generate_epoch_images_html(epochs_data: list, image_key: str, is_recon: bool = False) -> str:
    """Generate HTML for epoch images."""
    html_parts = []
    
    for epoch_data in epochs_data:
        epoch_num = epoch_data.get('epoch', 0) + 1
        images = epoch_data.get(image_key, [])
        
        if not images:
            continue
        
        gallery_class = 'recon-gallery' if is_recon else 'image-gallery'
        title = 'Generated' if not is_recon else 'Original | Reconstructed'
        
        html_parts.append(f'''
        <div class="epoch-section">
            <h3>Epoch {epoch_num} (Loss: {epoch_data.get('avg_loss', 0):.4f})</h3>
            <div class="{gallery_class}">
                {''.join(f'<img src="data:image/png;base64,{img}" alt="Epoch {epoch_num}">' for img in images)}
            </div>
        </div>
        ''')
    
    return '\n'.join(html_parts) if html_parts else '<p>No images available</p>'


def generate_html_report(metrics: dict, output_path: str) -> str:
    """Generate an HTML report; charts require the Chart.js CDN."""
    
    # Prepare data for charts
    epochs_data = metrics.get('epochs', [])
    batches_data = metrics.get('batches', [])
    
    # Extract loss curves
    epoch_nums = [e['epoch'] + 1 for e in epochs_data]
    epoch_losses = [e['avg_loss'] for e in epochs_data]
    epoch_times = [e['epoch_time_sec'] for e in epochs_data]
    
    # Batch-level data (sample every 10th for performance)
    sampled_batches = batches_data[::max(1, len(batches_data)//500)]
    batch_indices = list(range(len(sampled_batches)))
    batch_losses = [b.get('loss', 0) for b in sampled_batches]
    
    # GPU data - only include entries that have GPU metrics
    batches_with_gpu = [b for b in sampled_batches if b.get('gpu_memory_allocated_mb')]
    gpu_allocated = [b.get('gpu_memory_allocated_mb', 0) for b in batches_with_gpu] if batches_with_gpu else []
    gpu_reserved = [b.get('gpu_memory_reserved_mb', 0) for b in batches_with_gpu] if batches_with_gpu else []
    
    # Create HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Training Report - Diffusion Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8fafc;
            color: #1e293b;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{
            background: linear-gradient(135deg, #6366f1, #10b981);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
        }}
        header h1 {{ font-size: 2rem; margin-bottom: 10px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            font-size: 0.875rem;
            color: #64748b;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .stat-card .value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: #1e293b;
        }}
        .stat-card .trend {{
            font-size: 0.875rem;
            margin-top: 4px;
        }}
        .trend.down {{ color: #10b981; }}
        .trend.up {{ color: #ef4444; }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .chart-container h2 {{
            font-size: 1.25rem;
            margin-bottom: 15px;
            color: #1e293b;
        }}
        .chart-wrapper {{
            position: relative;
            height: 300px;
        }}
        .two-col {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }}
        .info-section {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .info-section h2 {{
            font-size: 1.25rem;
            margin-bottom: 15px;
        }}
        pre {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.875rem;
        }}
        .loss-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }}
        .loss-table th, .loss-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        .loss-table th {{
            background: #f8fafc;
            font-weight: 600;
            color: #64748b;
        }}
        .loss-table tr:hover {{ background: #f8fafc; }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .image-gallery img {{
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .recon-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .recon-gallery img {{
            width: 100%;
            aspect-ratio: 2;
            object-fit: cover;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .epoch-section {{
            margin-bottom: 30px;
            padding: 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .epoch-section h3 {{
            margin-bottom: 15px;
            color: #1e293b;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
        }}
        .epoch-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .epoch-images {{
            background: #f8fafc;
            padding: 15px;
            border-radius: 8px;
        }}
        .epoch-images h4 {{
            margin-bottom: 10px;
            font-size: 0.9rem;
            color: #64748b;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Training Report</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Training Time</h3>
                <div class="value">{metrics.get('total_time_sec', 0)/60:.1f} min</div>
            </div>
            <div class="stat-card">
                <h3>Final Loss</h3>
                <div class="value">{epoch_losses[-1]:.4f}</div>
                <div class="trend {'down' if epoch_losses[-1] < epoch_losses[0] else 'up'}">
                    {'▼' if epoch_losses[-1] < epoch_losses[0] else '▲'} {abs((epoch_losses[-1] - epoch_losses[0]) / epoch_losses[0] * 100):.1f}%
                </div>
            </div>
            <div class="stat-card">
                <h3>Initial Loss</h3>
                <div class="value">{epoch_losses[0]:.4f}</div>
            </div>
            <div class="stat-card">
                <h3>Epochs</h3>
                <div class="value">{len(epochs_data)}</div>
            </div>
            <div class="stat-card">
                <h3>Total Batches</h3>
                <div class="value">{metrics.get('total_batches', len(batches_data))}</div>
            </div>
            <div class="stat-card">
                <h3>CUDA Available</h3>
                <div class="value">{'Yes' if metrics['system_info']['cuda_available'] else 'No'}</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Loss Curve (Per Epoch)</h2>
            <div class="chart-wrapper">
                <canvas id="epochLossChart"></canvas>
            </div>
        </div>
        
        <div class="two-col">
            <div class="chart-container">
                <h2>Loss Curve (Per Batch)</h2>
                <div class="chart-wrapper">
                    <canvas id="batchLossChart"></canvas>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>Epoch Time</h2>
                <div class="chart-wrapper">
                    <canvas id="epochTimeChart"></canvas>
                </div>
            </div>
        </div>
        
        {f"""
        <div class="chart-container">
            <h2>GPU Memory Usage (MB)</h2>
            <div class="chart-wrapper">
                <canvas id="gpuMemoryChart"></canvas>
            </div>
        </div>
        """ if gpu_allocated and any(gpu_allocated) else ""}
        
        <div class="chart-container">
            <h2>Evaluation</h2>
            <p>Validation images: {metrics.get('validation_images', 0)}. Reconstruction quality is measured separately from generation quality.
            VQ-VAE requires a learned latent prior (for example, DiT); Gaussian vectors fed directly to its decoder are not a valid generation benchmark.</p>
        </div>
        
        <div class="chart-container">
            <h2>Reconstruction Comparison</h2>
            <p>Fixed images: originals first, followed by reconstructions in row order. Held-out images are used when validation is enabled; otherwise these are training previews.</p>
            {generate_epoch_images_html(epochs_data, 'recon_images', is_recon=True)}
        </div>
        
        <div class="info-section">
            <h2>System Information</h2>
            <pre>{json.dumps(metrics['system_info'], indent=2)}</pre>
        </div>
        
        <div class="info-section">
            <h2>Loss by Epoch</h2>
            <table class="loss-table">
                <thead>
                    <tr>
                        <th>Epoch</th>
                        <th>Avg Loss</th>
                        <th>Validation MSE [0,1]</th>
                        <th>Validation PSNR (dB)</th>
                        <th>Validation codebook perplexity</th>
                        <th>Time (s)</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f"""
                    <tr>
                        <td>{e['epoch'] + 1}</td>
                        <td>{e['avg_loss']:.6f}</td>
                        <td>{e.get('val_recon_mse', 'N/A')}</td>
                        <td>{e.get('val_psnr_db', 'N/A')}</td>
                        <td>{e.get('val_codebook_perplexity', 'N/A')}</td>
                        <td>{e['epoch_time_sec']:.2f}</td>
                    </tr>
                    """ for e in epochs_data)}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Epoch Loss Chart
        new Chart(document.getElementById('epochLossChart'), {{
            type: 'line',
            data: {{
                labels: {epoch_nums},
                datasets: [{{
                    label: 'Average Loss',
                    data: {epoch_losses},
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    tension: 0.3,
                    fill: true,
                    pointRadius: 4,
                    pointBackgroundColor: '#6366f1'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Loss' }}
                    }},
                    x: {{
                        title: {{ display: true, text: 'Epoch' }}
                    }}
                }}
            }}
        }});
        
        // Batch Loss Chart
        new Chart(document.getElementById('batchLossChart'), {{
            type: 'line',
            data: {{
                labels: {list(range(len(batch_losses)))},
                datasets: [{{
                    label: 'Batch Loss',
                    data: {batch_losses},
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.1,
                    pointRadius: 0,
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        title: {{ display: true, text: 'Loss' }}
                    }},
                    x: {{
                        display: false
                    }}
                }}
            }}
        }});
        
        // Epoch Time Chart
        new Chart(document.getElementById('epochTimeChart'), {{
            type: 'bar',
            data: {{
                labels: {epoch_nums},
                datasets: [{{
                    label: 'Time (s)',
                    data: {epoch_times},
                    backgroundColor: '#f59e0b'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Seconds' }}
                    }}
                }}
            }}
        }});
        
        {f"""
        // GPU Memory Chart
        new Chart(document.getElementById('gpuMemoryChart'), {{
            type: 'line',
            data: {{
                labels: {list(range(len(gpu_allocated)))},
                datasets: [
                    {{
                        label: 'Allocated',
                        data: {gpu_allocated},
                        borderColor: '#ef4444',
                        backgroundColor: 'transparent',
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 1
                    }},
                    {{
                        label: 'Reserved',
                        data: {gpu_reserved},
                        borderColor: '#f59e0b',
                        backgroundColor: 'transparent',
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'MB' }}
                    }},
                    x: {{
                        display: false
                    }}
                }}
            }}
        }});
        """ if gpu_allocated and len(gpu_allocated) > 0 else ""}
    </script>
</body>
</html>'''
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"HTML report saved to {output_path}")
    return output_path


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VAE/VQVAE with metrics tracking')
    parser.add_argument('--model', type=str, default='vae', choices=['vae', 'vqvae'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-limit', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--precision', choices=['auto', 'fp32', 'fp16', 'bf16'], default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--kld-beta', type=float, default=1.0)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--image-size', type=int, default=48)
    parser.add_argument('--latent-spatial-size', type=int, default=4)
    parser.add_argument('--data-path', default=None, help='Native uint8 N,H,W,3 .npy data')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--split-mode', choices=['random', 'validation_first'], default='random')
    parser.add_argument('--val-count', type=int, default=None)
    parser.add_argument('--vq-beta', type=float, default=1.0)
    parser.add_argument('--vq-recon-reduction', choices=['mean', 'sum'], default='mean')
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--num-embeddings', type=int, default=512)
    parser.add_argument('--commitment-cost', type=float, default=0.25)
    parser.add_argument('--distance-chunk-size', type=int, default=4096)
    parser.add_argument('--kl-warmup-epochs', type=int, default=0)
    parser.add_argument('--init-checkpoint', default=None, help='Strict weight transfer into a new run; not optimizer resume')
    
    args = parser.parse_args()
    
    try:
        result = train_vae_cmdline(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_limit=args.data_limit,
            lr=args.lr,
            use_bfloat16=args.bf16,
            latent_dim=args.latent_dim,
            device=args.device,
            precision=args.precision, seed=args.seed, kld_beta=args.kld_beta, output_dir=args.output_dir,
            image_size=args.image_size, latent_spatial_size=args.latent_spatial_size,
            data_path=args.data_path, split_mode=args.split_mode, val_count=args.val_count,
            validation_fraction=args.validation_fraction, vq_beta=args.vq_beta,
            vq_recon_reduction=args.vq_recon_reduction, embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings, commitment_cost=args.commitment_cost,
            distance_chunk_size=args.distance_chunk_size, kl_warmup_epochs=args.kl_warmup_epochs,
            init_checkpoint=args.init_checkpoint,
        )
        
        print(f"\n{'='*60}")
        print("Training completed successfully!")
        print(f"Model: {result['model_path']}")
        print(f"Metrics: {result['metrics_path']}")
        print(f"Report: {result['html_path']}")
        print(f"{'='*60}")
        
        # Print summary
        metrics = result['metrics']
        epochs = metrics['epochs']
        if len(epochs) >= 2:
            initial_loss = epochs[0]['avg_loss']
            final_loss = epochs[-1]['avg_loss']
            reduction = (initial_loss - final_loss) / initial_loss * 100
            
            print(f"\nLoss Summary:")
            print(f"  Initial: {initial_loss:.6f}")
            print(f"  Final:   {final_loss:.6f}")
            print(f"  Reduction: {reduction:.1f}%")
            
            if reduction > 10:
                print("  ✓ Loss decreased significantly!")
            elif reduction > 0:
                print("  ✓ Loss decreased moderately")
            else:
                print("  ⚠ Loss did not decrease - may need more epochs or higher learning rate")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("Training failed!")
        print(f"Error: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
