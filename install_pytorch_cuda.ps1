# PowerShell script to install PyTorch with CUDA support
# For NVIDIA GeForce GTX 1650 Ti with CUDA 13.0

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "Installing PyTorch with CUDA Support" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan

Write-Host "`nYour GPU: NVIDIA GeForce GTX 1650 Ti (CUDA 13.0)" -ForegroundColor Green

Write-Host "`nStep 1: Uninstalling CPU-only PyTorch..." -ForegroundColor Cyan
pip uninstall -y torch torchvision torchaudio

Write-Host "`nStep 2: Installing PyTorch with CUDA 12.4 support..." -ForegroundColor Cyan
Write-Host "(CUDA 12.4 is compatible with your CUDA 13.0 driver)" -ForegroundColor Gray
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host "`nStep 3: Verifying installation..." -ForegroundColor Cyan
python check_cuda_setup.py

Write-Host "`nInstallation complete!" -ForegroundColor Green
Write-Host "You can now train your model with GPU acceleration." -ForegroundColor Green
