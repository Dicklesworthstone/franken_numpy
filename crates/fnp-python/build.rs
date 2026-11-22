use std::path::PathBuf;
use std::process::Command;

fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn build_python() -> String {
    if let Some(py) = env_nonempty("PYO3_PYTHON") {
        return py;
    }
    if let Some(venv) = env_nonempty("VIRTUAL_ENV") {
        for candidate in ["bin/python", "bin/python3", "Scripts/python.exe"] {
            let path = PathBuf::from(&venv).join(candidate);
            if path.is_file() {
                return path.to_string_lossy().into_owned();
            }
        }
    }
    "python3".to_string()
}

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");

    let python = build_python();
    if let Ok(output) = Command::new(&python)
        .args([
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
        ])
        .output()
        && output.status.success()
    {
        let xy = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !xy.is_empty() && xy.split('.').count() == 2 {
            println!("cargo:rustc-env=FNP_BUILT_PYTHON_XY={xy}");
        }
    }
}
