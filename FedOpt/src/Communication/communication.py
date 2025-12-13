#!/usr/bin/env python3
"""
Backward Compatibility Module

This module provides backward compatible imports for code that uses:
    from Communication.communication import CommunicationManager
    from Communication.communication import json_to_tensor_list, ...

All functionality is re-exported from communication_manager.py
"""

# Re-export everything from communication_manager for backward compatibility
from .communication_manager import (
    CommunicationManager,
    create_typed_message,
    recv_all,
    state_dict_to_json,
    json_to_state_dict,
    tensor_list_to_json,
    json_to_tensor_list,
    check_model_size,
    update_client_variables,
)

# Also expose logging and other utilities that might be expected
import logging
logger = logging.getLogger("FedOpt")

__all__ = [
    'CommunicationManager',
    'create_typed_message',
    'recv_all',
    'state_dict_to_json',
    'json_to_state_dict',
    'tensor_list_to_json',
    'json_to_tensor_list',
    'check_model_size',
    'update_client_variables',
    'logger',
]
