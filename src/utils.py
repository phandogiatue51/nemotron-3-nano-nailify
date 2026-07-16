# src/utils.py

import json
import yaml
from typing import Dict, Any
import logging
from datetime import datetime

def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nailify_rag.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_customer_data(data: Dict) -> bool:
    """Validate customer data structure"""
    required_fields = ['userId', 'email', 'firstName', 'lastName']
    return all(field in data for field in required_fields)

def format_recommendation_for_api(result: Dict) -> Dict:
    """Format recommendation for API response"""
    return {
        "status": "success",
        "data": {
            "recommendation": result,
            "timestamp": datetime.now().isoformat()
        }
    }