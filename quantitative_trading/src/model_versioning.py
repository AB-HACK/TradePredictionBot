"""
Model Versioning Module
Handles model version management and tracking
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """Manage model versions and metadata"""
    
    @staticmethod
    def generate_version(model_name: str, ticker: str, metadata: Dict) -> str:
        """Generate version string for model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Create hash from metadata
        metadata_str = json.dumps(metadata, sort_keys=True)
        hash_obj = hashlib.md5(metadata_str.encode())
        hash_short = hash_obj.hexdigest()[:8]
        return f"{ticker}_{model_name}_{timestamp}_{hash_short}"
    
    @staticmethod
    def save_version_info(version: str, metadata: Dict, save_dir: str, keep_versions: int = 5, config=None):
        """Save version information"""
        version_file = os.path.join(save_dir, 'versions.json')
        versions = []
        
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r') as f:
                    versions = json.load(f)
            except:
                versions = []
        
        versions.append({
            'version': version,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata
        })
        
        # Keep only last N versions
        if config:
            keep_versions = config.get('model_versioning.keep_versions', keep_versions)
        versions = versions[-keep_versions:]
        
        with open(version_file, 'w') as f:
            json.dump(versions, f, indent=2)
        
        logger.info(f"Saved version info: {version}")

