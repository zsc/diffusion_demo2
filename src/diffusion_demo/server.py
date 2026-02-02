"""Web server entry point."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, render_template, send_from_directory

from .api import api_bp
from .utils.torch_env import get_env_config


def create_app() -> Flask:
    """Create Flask application."""
    app = Flask(
        __name__,
        template_folder='ui/templates',
        static_folder='ui/static',
    )
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Ensure required directories exist
    for dir_name in ['cache', 'checkpoints', 'runs', 'results', 'var/jobs']:
        os.makedirs(dir_name, exist_ok=True)
    
    @app.route('/')
    def index():
        """Serve main page."""
        return render_template('index.html')
    
    return app


def main():
    """Main entry point."""
    config = get_env_config()
    
    app = create_app()
    
    print(f"Starting Diffusion Demo server on http://{config['host']}:{config['port']}")
    print(f"Press Ctrl+C to stop")
    
    app.run(
        host=config['host'],
        port=config['port'],
        debug=False,
        threaded=True,
    )


if __name__ == '__main__':
    main()
