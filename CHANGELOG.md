# Changelog

All notable changes to this project will be documented in this file.

## [1.1.2] - 2025-08-22

### Fixed
- Model generation parameters (`do_sample=True` for temperature/top_p)
- Explicit GPU memory management and device handling
- Added detailed GPU information logging
- Improved inference debugging with status messages

## [1.1.1] - 2025-08-22

### Added
- Debugging logs for input validation
- Test scripts for API troubleshooting

## [1.1.0] - 2025-08-21

### Added
- Initial RunPod serverless deployment configuration
- Docker container setup with NVIDIA CUDA support
- Python environment configuration
- Hugging Face model integration
- Storage monitoring functionality
- Network volume support (80GB in US-WA-1)
- GitHub Actions CI/CD pipeline

### Changed
- Base image to `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` for stability
- Python version to 3.10 (Ubuntu 22.04 default)
- Environment variables for non-interactive installation

### Fixed
- Docker image tag compatibility issues
- Python installation via proper package management
- tzdata interactive prompt during Docker build
- Storage path configuration for RunPod volumes

### Technical Details
- **Base Image**: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
- **Python**: 3.10
- **CUDA**: 12.1.1
- **cuDNN**: 8
- **Storage**: 80GB Network Volume at /runpod-volume
- **Region**: US-WA-1