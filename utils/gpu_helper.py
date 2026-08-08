import os
import subprocess
import shutil
import platform
import psutil

AMD_GFX_TARGETS = {
    "Auto-Detect / Default": "",
    "RDNA 2 - RX 6000 Series (gfx1030)": "10.3.0",
    "RDNA 3 - RX 7000 Series (gfx1100)": "11.0.0",
    "RDNA 1 - RX 5000 Series (gfx1010)": "10.1.0",
    "Vega 56 / 64 / VII (gfx900/906)": "9.0.0",
    "MI100 / MI200 Series (gfx90a)": "9.0.a",
    "MI300 Series (gfx942)": "9.4.2"
}

def detect_gpu_hardware() -> dict:
    """
    Detects system GPU hardware vendor, available VRAM, driver status, and ROCm/CUDA availability.
    Returns structured hardware diagnostic dictionary.
    """
    system_info = {
        "vendor": "CPU",
        "gpu_name": "CPU Fallback",
        "vram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "vram_free_gb": round(psutil.virtual_memory().available / (1024**3), 1),
        "driver_version": "N/A",
        "has_cuda": False,
        "has_rocm": False,
        "has_vulkan": False,
        "recommended_backend": "CPU",
        "gfx_version_default": ""
    }

    # 1. Check NVIDIA CUDA via nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            res = subprocess.run(
                [nvidia_smi, "--query-gpu=gpu_name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split("\n")[0].split(",")]
                if len(parts) >= 4:
                    system_info["vendor"] = "NVIDIA"
                    system_info["gpu_name"] = parts[0]
                    system_info["vram_total_gb"] = round(float(parts[1]) / 1024.0, 1)
                    system_info["vram_free_gb"] = round(float(parts[2]) / 1024.0, 1)
                    system_info["driver_version"] = f"NVIDIA {parts[3]}"
                    system_info["has_cuda"] = True
                    system_info["recommended_backend"] = "CUDA"
                    return system_info
        except Exception:
            pass

    # 2. Check AMD ROCm / HIP via rocm-smi or rocminfo or /dev/kfd
    rocm_smi = shutil.which("rocm-smi")
    has_kfd = os.path.exists("/dev/kfd")
    
    if rocm_smi or has_kfd:
        system_info["has_rocm"] = True
        system_info["vendor"] = "AMD"
        system_info["recommended_backend"] = "ROCm (HIP)"

        if rocm_smi:
            try:
                res = subprocess.run(
                    [rocm_smi, "--showname", "--showdriverversion"],
                    capture_output=True, text=True, timeout=5
                )
                if res.returncode == 0 and res.stdout.strip():
                    lines = res.stdout.strip().splitlines()
                    for line in lines:
                        if "Card series" in line or "Card model" in line:
                            system_info["gpu_name"] = line.split(":")[-1].strip()
                        if "Driver version" in line:
                            system_info["driver_version"] = f"ROCm {line.split(':')[-1].strip()}"
            except Exception:
                pass

        # Try extracting VRAM info via rocm-smi
        if rocm_smi:
            try:
                res_mem = subprocess.run(
                    [rocm_smi, "--showmeminfo", "vram"],
                    capture_output=True, text=True, timeout=5
                )
                if res_mem.returncode == 0 and res_mem.stdout.strip():
                    for line in res_mem.stdout.splitlines():
                        if "VRAM Total Memory" in line:
                            val = line.split(":")[-1].strip()
                            if val.isdigit():
                                system_info["vram_total_gb"] = round(int(val) / (1024**3), 1)
                        if "VRAM Total Used" in line:
                            val = line.split(":")[-1].strip()
                            if val.isdigit():
                                total = system_info["vram_total_gb"]
                                used = round(int(val) / (1024**3), 1)
                                system_info["vram_free_gb"] = max(0.0, round(total - used, 1))
            except Exception:
                pass

    # 3. Check AMD GPU via lspci if rocm-smi output was generic
    if system_info["gpu_name"] == "CPU Fallback":
        lspci = shutil.which("lspci")
        if lspci:
            try:
                res_pci = subprocess.run([lspci], capture_output=True, text=True, timeout=5)
                if res_pci.returncode == 0:
                    for line in res_pci.stdout.splitlines():
                        if "VGA" in line or "3D" in line:
                            if "AMD" in line or "Radeon" in line or "Advanced Micro Devices" in line:
                                system_info["vendor"] = "AMD"
                                clean_name = line.split(":")[-1].strip()
                                system_info["gpu_name"] = clean_name
                                if system_info["recommended_backend"] == "CPU":
                                    system_info["recommended_backend"] = "Vulkan"
                                break
            except Exception:
                pass

    # 4. Check Vulkan availability
    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo:
        system_info["has_vulkan"] = True

    return system_info

def configure_amd_gpu_env(gfx_version: str = "", hip_device: str = "0", rocm_path: str = "/opt/rocm") -> dict:
    """
    Sets environment variables for AMD ROCm / HIP acceleration.
    """
    env_updates = {}
    if gfx_version.strip():
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = gfx_version.strip()
        env_updates["HSA_OVERRIDE_GFX_VERSION"] = gfx_version.strip()
    elif "HSA_OVERRIDE_GFX_VERSION" in os.environ:
        del os.environ["HSA_OVERRIDE_GFX_VERSION"]

    os.environ["HIP_VISIBLE_DEVICES"] = str(hip_device)
    env_updates["HIP_VISIBLE_DEVICES"] = str(hip_device)

    if os.path.exists(rocm_path):
        os.environ["ROCM_PATH"] = rocm_path
        env_updates["ROCM_PATH"] = rocm_path

    return env_updates

if __name__ == "__main__":
    hw = detect_gpu_hardware()
    print("✓ Hardware Detection Results:")
    for k, v in hw.items():
        print(f"  {k}: {v}")
