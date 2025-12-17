//! CLI tool for serializing crowd-pilot IDE interaction data.
//!
//! This tool processes CSV session files and outputs JSONL format suitable for
//! NeMo SFT training. It uses an embedded Python interpreter to load HuggingFace
//! tokenizers for accurate token counting.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use clap::Parser;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crowd_pilot_serializer_core::{
    pipeline::{PipelineConfig, PipelineResult},
    process_all_sessions, write_jsonl_output, Tokenizer,
};

/// Serialize crowd-pilot CSV sessions to NeMo JSONL format.
#[derive(Parser, Debug)]
#[command(name = "crowd-pilot-serialize")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Root directory containing CSV session files
    #[arg(long)]
    csv_root: PathBuf,

    /// Output directory for JSONL files
    #[arg(long)]
    output_dir: PathBuf,

    /// HuggingFace tokenizer model name or path
    #[arg(long)]
    tokenizer: String,

    /// Maximum tokens per conversation chunk
    #[arg(long, default_value = "8192")]
    max_tokens_per_conversation: usize,

    /// Maximum tokens per message
    #[arg(long, default_value = "2048")]
    max_tokens_per_message: usize,

    /// Minimum messages required to keep a conversation
    #[arg(long, default_value = "5")]
    min_conversation_messages: usize,

    /// Viewport radius (lines above/below cursor)
    #[arg(long, default_value = "10")]
    viewport_radius: usize,

    /// Coalesce radius for grouping nearby edits
    #[arg(long, default_value = "5")]
    coalesce_radius: usize,

    /// Fraction of sessions for validation (0.0-1.0)
    #[arg(long, default_value = "0.1")]
    val_ratio: f64,

    /// Custom system prompt (optional)
    #[arg(long)]
    system_prompt: Option<String>,
}

const DEFAULT_SYSTEM_PROMPT: &str = r#"You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Format your response as shown in <format_example>.

<format_example>
```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."#;

/// Wrapper around Python tokenizer for exact token counting and truncation.
struct PythonTokenizer {
    tokenizer: Py<PyAny>,
}

impl PythonTokenizer {
    /// Load a HuggingFace tokenizer.
    fn load(model_name: &str) -> PyResult<Self> {
        Python::with_gil(|py| {
            let transformers = PyModule::import(py, "transformers")?;
            let auto_tokenizer = transformers.getattr("AutoTokenizer")?;
            let tokenizer = auto_tokenizer.call_method1("from_pretrained", (model_name,))?;
            Ok(Self {
                tokenizer: tokenizer.into(),
            })
        })
    }
}

impl Tokenizer for PythonTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        Python::with_gil(|py| {
            let tokenizer = self.tokenizer.as_ref(py);
            let tokens = tokenizer
                .call_method1("encode", (text,))
                .expect("Failed to encode text with tokenizer");
            tokens.len().unwrap()
        })
    }

    fn truncate_to_max_tokens(&self, text: &str, max_tokens: usize) -> String {
        Python::with_gil(|py| {
            let tokenizer = self.tokenizer.as_ref(py);
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("max_length", max_tokens).unwrap();
            kwargs.set_item("truncation", true).unwrap();
            
            let tokens = tokenizer
                .call_method("encode", (text,), Some(kwargs))
                .expect("Failed to encode text with tokenizer");
            
            tokenizer
                .call_method1("decode", (tokens,))
                .expect("Failed to decode tokens")
                .extract()
                .unwrap()
        })
    }
}

/// Thread-safe wrapper around PythonTokenizer.
///
/// Uses a Mutex to ensure only one thread accesses the Python tokenizer at a time.
/// This is necessary because `Py<PyAny>` is `Send` but not `Sync`.
/// 
/// Note: Python's GIL already serializes access, so this doesn't add overhead.
struct ThreadSafeTokenizer {
    inner: Mutex<PythonTokenizer>,
}

impl ThreadSafeTokenizer {
    fn new(tokenizer: PythonTokenizer) -> Self {
        Self {
            inner: Mutex::new(tokenizer),
        }
    }
}

impl Tokenizer for ThreadSafeTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        let tokenizer = self.inner.lock().expect("Tokenizer mutex poisoned");
        tokenizer.count_tokens(text)
    }

    fn truncate_to_max_tokens(&self, text: &str, max_tokens: usize) -> String {
        let tokenizer = self.inner.lock().expect("Tokenizer mutex poisoned");
        tokenizer.truncate_to_max_tokens(text, max_tokens)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading tokenizer from {}...", args.tokenizer);
    let tokenizer = PythonTokenizer::load(&args.tokenizer)?;
    let tokenizer = Arc::new(ThreadSafeTokenizer::new(tokenizer));

    let config = PipelineConfig {
        max_tokens_per_conversation: args.max_tokens_per_conversation,
        max_tokens_per_message: args.max_tokens_per_message,
        min_conversation_messages: args.min_conversation_messages,
        viewport_radius: args.viewport_radius,
        coalesce_radius: args.coalesce_radius,
        val_ratio: args.val_ratio,
    };

    println!("Processing CSV files from {:?}...", args.csv_root);
    let session_results = process_all_sessions(
        &args.csv_root,
        tokenizer.as_ref(),
        &config,
    )?;

    let total_sessions = session_results.len();
    println!("Processed {} sessions", total_sessions);

    let system_prompt = args.system_prompt.as_deref().unwrap_or(DEFAULT_SYSTEM_PROMPT);

    println!("Writing output to {:?}...", args.output_dir);
    let result: PipelineResult = write_jsonl_output(
        session_results,
        &args.output_dir,
        args.val_ratio,
        system_prompt,
    )?;

    let metadata_path = args.output_dir.join("metadata.json");
    let metadata = serde_json::json!({
        "config": {
            "csv_root": args.csv_root.to_string_lossy(),
            "output_dir": args.output_dir.to_string_lossy(),
            "tokenizer": args.tokenizer,
            "max_tokens_per_conversation": args.max_tokens_per_conversation,
            "max_tokens_per_message": args.max_tokens_per_message,
            "min_conversation_messages": args.min_conversation_messages,
            "viewport_radius": args.viewport_radius,
            "coalesce_radius": args.coalesce_radius,
            "val_ratio": args.val_ratio,
        },
        "counts": {
            "total_sessions": result.total_sessions,
            "total_conversations": result.total_conversations,
            "train_conversations": result.train_conversations,
            "val_conversations": result.val_conversations,
        },
        "stats": {
            "total_messages": result.total_messages,
            "total_tokens": result.total_tokens,
            "avg_messages_per_conversation": if result.total_conversations > 0 {
                result.total_messages as f64 / result.total_conversations as f64
            } else {
                0.0
            },
            "avg_tokens_per_conversation": if result.total_conversations > 0 {
                result.total_tokens as f64 / result.total_conversations as f64
            } else {
                0.0
            },
        },
        "files": {
            "train_path": args.output_dir.join("training.jsonl").to_string_lossy(),
            "val_path": args.output_dir.join("validation.jsonl").to_string_lossy(),
        },
    });
    std::fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;

    println!("\n[summary]");
    println!("  Total sessions processed: {}", result.total_sessions);
    println!("  Train conversations: {}", result.train_conversations);
    println!("  Val conversations: {}", result.val_conversations);
    println!("  Total messages: {}", result.total_messages);
    println!("  Total tokens: {}", result.total_tokens);
    println!("  Output: {:?}/{{training,validation}}.jsonl", args.output_dir);
    println!("  Metadata: {:?}", metadata_path);

    Ok(())
}

