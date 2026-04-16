# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Triton Fused Operators, please follow responsible disclosure:

### DO NOT:
- Open a public issue describing the vulnerability
- Submit a pull request with the fix before coordinating with maintainers

### DO:
1. Email maintainers privately with details
2. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if available)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week  
- **Fix timeline**: Depends on severity
- **Credit**: Given to reporter (if desired) upon resolution

## Security Considerations

This library provides GPU kernels for PyTorch. Security considerations include:

- **Numerical Stability**: FP8 quantization may affect model accuracy
- **Memory Safety**: Triton kernels operate close to hardware
- **API Compatibility**: Breaking changes are documented

If you encounter numerical instability or undefined behavior, please report it as a security concern.
