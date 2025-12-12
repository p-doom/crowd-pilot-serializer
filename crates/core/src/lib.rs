//! Core serialization logic for crowd-pilot IDE interaction data.
//!
//! This crate provides the `ConversationStateManager` which converts IDE events
//! (tab switches, edits, terminal commands, etc.) into conversation format
//! suitable for training language models.

/// Trait for tokenization operations.
/// 
/// Implementors provide token counting and truncation capabilities.
/// For exact tokenization (preprocessing), use a real tokenizer.
/// For approximate tokenization (runtime), use character-based estimation.
pub trait Tokenizer {
    /// Count the number of tokens in the given text.
    fn count_tokens(&self, text: &str) -> usize;
    
    /// Truncate text to at most `max_tokens` tokens.
    /// Returns the truncated text.
    fn truncate_to_max_tokens(&self, text: &str, max_tokens: usize) -> String;
}

// Blanket implementation for references to Tokenizers
impl<T: Tokenizer + ?Sized> Tokenizer for &T {
    fn count_tokens(&self, text: &str) -> usize {
        (*self).count_tokens(text)
    }
    
    fn truncate_to_max_tokens(&self, text: &str, max_tokens: usize) -> String {
        (*self).truncate_to_max_tokens(text, max_tokens)
    }
}

mod conversation;
mod diff;
mod helpers;
pub mod pipeline;

pub use conversation::{ConversationMessage, ConversationStateManager, ConversationStateManagerConfig, FinalizedConversation};
pub use pipeline::{
    discover_csv_files, process_all_sessions, process_session, write_jsonl_output,
    NemoMessage, NemoRecord, PipelineConfig, PipelineResult, SessionResult,
};
pub use diff::{compute_changed_block_lines, ChangedBlock};
pub use helpers::{
    apply_backspaces, apply_change, clean_text, escape_single_quotes_for_sed, fenced_block,
    line_numbered_output, normalize_terminal_output, serialize_compute_viewport, Viewport,
};

/// Default viewport radius (lines above/below cursor to show)
pub const VIEWPORT_RADIUS: usize = 10;

/// Default coalesce radius for grouping nearby edits
pub const COALESCE_RADIUS: usize = 5;

/// Default maximum tokens per message (approximate)
pub const MAX_TOKENS_PER_MESSAGE: usize = 2048;

/// Default maximum tokens per terminal output
pub const MAX_TOKENS_PER_TERMINAL_OUTPUT: usize = 256;

