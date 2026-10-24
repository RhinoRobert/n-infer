//! What this machine can do, and a guard against running a binary built for a newer CPU.

use llama_cpp_2::list_llama_ggml_backend_devices;
use napi_derive::napi;

use crate::backend;
use crate::error::{err, Result};

const MIB: f64 = 1024.0 * 1024.0;

#[napi(object)]
pub struct DeviceInfo {
    pub name: String,
    pub description: String,
    /// "CUDA", "Vulkan", "CPU", ...
    pub backend: String,
    pub kind: String,
    pub total_mib: f64,
    pub free_mib: f64,
}

#[napi(object)]
pub struct HardwareInfo {
    pub physical_cores: u32,
    pub logical_cores: u32,
    /// Accelerator compiled in: "cuda", "vulkan" or "cpu".
    pub build_backend: String,
    /// CPU instruction sets this CPU supports (x86 only).
    pub cpu_features: Vec<String>,
    /// Instruction sets the binary was compiled to require (x86 only).
    pub compiled_for: Vec<String>,
    /// llama.cpp's own report of the CPU features it was compiled to use.
    pub system_info: String,
    pub devices: Vec<DeviceInfo>,
}

fn llama_system_info() -> String {
    // SAFETY: returns a pointer to a static NUL-terminated string owned by llama.cpp.
    let ptr = unsafe { llama_cpp_sys_2::llama_print_system_info() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(ptr) }.to_string_lossy().trim().to_string()
}

#[cfg(target_arch = "x86_64")]
macro_rules! x86_features {
    ($probe:ident) => {{
        let mut found = Vec::new();
        for (name, present) in [
            ("sse4.2", $probe!("sse4.2")),
            ("avx", $probe!("avx")),
            ("avx2", $probe!("avx2")),
            ("fma", $probe!("fma")),
            ("f16c", $probe!("f16c")),
            ("bmi2", $probe!("bmi2")),
            ("avx512f", $probe!("avx512f")),
        ] {
            if present {
                found.push(name.to_string());
            }
        }
        found
    }};
}

#[cfg(target_arch = "x86_64")]
fn detected_features() -> Vec<String> {
    macro_rules! runtime {
        ($f:tt) => {
            std::arch::is_x86_feature_detected!($f)
        };
    }
    x86_features!(runtime)
}

#[cfg(target_arch = "x86_64")]
fn compiled_features() -> Vec<String> {
    macro_rules! compiled {
        ($f:tt) => {
            cfg!(target_feature = $f)
        };
    }
    x86_features!(compiled)
}

#[cfg(not(target_arch = "x86_64"))]
fn detected_features() -> Vec<String> {
    Vec::new()
}

#[cfg(not(target_arch = "x86_64"))]
fn compiled_features() -> Vec<String> {
    Vec::new()
}

/// A binary compiled with `target-cpu=native` (or a newer baseline) faults with an illegal
/// instruction on older CPUs. Fail with a readable error before any native code runs.
pub fn ensure_cpu_supported() -> Result<()> {
    let detected = detected_features();
    let missing: Vec<String> = compiled_features()
        .into_iter()
        .filter(|f| !detected.contains(f))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    Err(err(format!(
        "this n-infer binary was built for a newer CPU (needs {}). \
         Rebuild on this machine with `npm run build`, or use `npm run build:baseline`.",
        missing.join(", ")
    )))
}

pub fn info() -> Result<HardwareInfo> {
    backend::get(false)?;
    let devices = list_llama_ggml_backend_devices()
        .into_iter()
        .map(|d| DeviceInfo {
            name: d.name,
            description: d.description,
            backend: d.backend,
            kind: format!("{:?}", d.device_type),
            total_mib: d.memory_total as f64 / MIB,
            free_mib: d.memory_free as f64 / MIB,
        })
        .collect();
    Ok(HardwareInfo {
        physical_cores: num_cpus::get_physical() as u32,
        logical_cores: num_cpus::get() as u32,
        build_backend: crate::binding::build_backend().to_string(),
        cpu_features: detected_features(),
        compiled_for: compiled_features(),
        system_info: llama_system_info(),
        devices,
    })
}
