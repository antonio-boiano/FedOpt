#!/usr/bin/env python3
"""
gRPC Generated Modules

This folder should contain the generated gRPC files:
- FedOpt_pb2.py
- FedOpt_pb2_grpc.py

To generate these files, run from the directory containing FedOpt.proto:
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. FedOpt.proto

Then copy the generated files to this directory.
"""

# Try to import the generated modules if they exist
try:
    from .FedOpt_pb2 import *
    from .FedOpt_pb2_grpc import *
except ImportError:
    pass  # Files not yet generated/copied
