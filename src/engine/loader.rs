//! Everything that happens before llama.cpp opens a model file: checking that it is really a
//! GGUF file, optionally verifying its hash, and choosing how much of it lives on the GPU.

use std::ffi::CString;
use std::fs::File;
use std::io::Read;
use std::pin::Pin;
use std::sync::Mutex;

use llama_cpp_2::model::params::LlamaModelParams;
use sha2::{Digest, Sha256};

use crate::error::{err, Result};
use crate::options::ValidatedModelConfig;
use crate::tuning::Tuning;

/// Smallest file that can hold a GGUF header.
const MIN_GGUF_BYTES: u64 = 64;
/// VRAM left untouched per device when fitting a model to the GPU.
const FIT_MARGIN_BYTES: usize = 1024 * 1024 * 1024;
/// Context size the fitter may shrink to before it starts moving layers to the CPU.
const FIT_MIN_CONTEXT: u32 = 2048;
/// llama.cpp's fitter mutates global logger state, so only one may run at a time.
static FIT_LOCK: Mutex<()> = Mutex::new(());

/// Checks that `path` is a regular file starting with the GGUF magic and a known version,
/// before llama.cpp's C++ parser sees it.
pub fn validate_gguf_file(path: &str) -> Result<()> {
    let mut file = File::open(path).map_err(|e| err(format!("cannot open model '{path}': {e}")))?;
    let meta = file.metadata().map_err(|e| err(format!("cannot stat model '{path}': {e}")))?;
    if !meta.is_file() {
        return Err(err(format!("model path '{path}' is not a regular file")));
    }
    if meta.len() < MIN_GGUF_BYTES {
        return Err(err(format!("model file '{path}' is too small to be a GGUF file")));
    }
    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .map_err(|e| err(format!("cannot read header of '{path}': {e}")))?;
    check_gguf_header(&header, path)
}

fn check_gguf_header(header: &[u8; 8], path: &str) -> Result<()> {
    if &header[0..4] != b"GGUF" {
        return Err(err(format!("invalid model file '{path}': missing GGUF magic header")));
    }
    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if !(1..=3).contains(&version) {
        return Err(err(format!("unsupported GGUF format version ({version}) in '{path}'")));
    }
    Ok(())
}

/// Streams the file through SHA-256 and compares it with `expected` (lowercase hex).
pub fn verify_sha256(path: &str, expected: &str) -> Result<()> {
    let mut file = File::open(path).map_err(|e| err(format!("cannot open model '{path}': {e}")))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|e| err(format!("cannot read '{path}': {e}")))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        return Err(err(format!(
            "model file '{path}' does not match modelSha256 (expected {expected}, found {actual})"
        )));
    }
    Ok(())
}

/// Model params for the requested offload. With no explicit layer count (the default on GPU
/// builds) llama.cpp's fitter splits the model between VRAM and RAM for this machine.
pub fn model_params(cfg: &ValidatedModelConfig, tuning: &Tuning) -> Pin<Box<LlamaModelParams>> {
    match cfg.gpu_layers {
        Some(layers) => Box::pin(LlamaModelParams::default().with_n_gpu_layers(layers)),
        None => fit_to_vram(cfg, tuning)
            .unwrap_or_else(|| Box::pin(LlamaModelParams::default().with_n_gpu_layers(0))),
    }
}

fn fit_to_vram(cfg: &ValidatedModelConfig, tuning: &Tuning) -> Option<Pin<Box<LlamaModelParams>>> {
    let path = CString::new(cfg.model_path.as_str()).ok()?;
    let mut params = Box::pin(LlamaModelParams::default());
    let mut cparams = tuning.context_params(cfg.context_size);
    let mut margins = vec![FIT_MARGIN_BYTES; llama_cpp_2::max_devices().max(1)];
    let log_level = if cfg.verbose { 2 } else { 4 };

    let _guard = FIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let fitted = params.as_mut().fit_params(
        &path,
        &mut cparams,
        &mut margins,
        FIT_MIN_CONTEXT.min(cfg.context_size),
        log_level,
    );
    match fitted {
        Ok(_) => Some(params),
        Err(e) => {
            if cfg.verbose {
                eprintln!("n-infer: could not fit the model to VRAM ({e}); using the CPU");
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file(name: &str, bytes: &[u8]) -> String {
        let path = std::env::temp_dir().join(format!("n-infer-test-{}-{name}", std::process::id()));
        File::create(&path).unwrap().write_all(bytes).unwrap();
        path.to_string_lossy().into_owned()
    }

    fn gguf_bytes(version: u32) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.resize(128, 0);
        bytes
    }

    #[test]
    fn accepts_a_valid_header() {
        let path = temp_file("ok.gguf", &gguf_bytes(3));
        assert!(validate_gguf_file(&path).is_ok());
    }

    #[test]
    fn rejects_wrong_magic_and_unknown_versions() {
        let mut bad_magic = gguf_bytes(3);
        bad_magic[0] = b'X';
        assert!(validate_gguf_file(&temp_file("magic.gguf", &bad_magic)).is_err());
        assert!(validate_gguf_file(&temp_file("v9.gguf", &gguf_bytes(9))).is_err());
        assert!(validate_gguf_file(&temp_file("v0.gguf", &gguf_bytes(0))).is_err());
    }

    #[test]
    fn rejects_tiny_files_directories_and_missing_paths() {
        assert!(validate_gguf_file(&temp_file("tiny.gguf", b"GGUF")).is_err());
        assert!(validate_gguf_file(&std::env::temp_dir().to_string_lossy()).is_err());
        assert!(validate_gguf_file("definitely/not/here.gguf").is_err());
    }

    #[test]
    fn verifies_sha256() {
        let path = temp_file("hash.bin", b"abc");
        let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(verify_sha256(&path, abc).is_ok());
        assert!(verify_sha256(&path, &"0".repeat(64)).is_err());
    }
}
