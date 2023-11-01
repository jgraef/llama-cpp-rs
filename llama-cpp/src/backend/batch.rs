//! Batched decoding

use std::slice;

use super::{
    Pos,
    SeqId,
    Token,
};

/// llama batch
pub struct Batch {
    pub(super) data: llama_cpp_sys::llama_batch,
    n_embd: usize,
    tokens_buf_size: usize,
    n_seq_max: usize,
}

unsafe impl Send for Batch {}

impl Batch {
    /// Create new batch that will accept tokens
    ///
    /// # Panics
    ///
    /// Panics if `n_tokens` is 0.
    pub fn new_tokens(n_tokens: usize, n_seq_max: usize) -> Self {
        Self::new(n_tokens, 0, n_seq_max)
    }

    fn new(n_tokens: usize, n_embd: usize, n_seq_max: usize) -> Self {
        assert!(n_tokens > 0);

        tracing::trace!(n_tokens, n_seq_max, "calling llama_batch_init");
        let data = unsafe {
            // for `embd` we need to pass 0, otherwise data.tokens is not allocated and thus
            // we are unsafe. we should write a separate `Batch` for embedding.
            llama_cpp_sys::llama_batch_init(n_tokens as _, n_embd as _, n_seq_max as _)
        };

        Self {
            data,
            n_embd,
            tokens_buf_size: n_tokens,
            n_seq_max,
        }
    }

    /// Returns the size of the token buffer.
    pub fn size(&self) -> usize {
        self.tokens_buf_size
    }

    /// Returns the embedding size, or 0 if this buffer can't accept embeddings.
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// Returns the max number of sequences.
    pub fn n_seq_max(&self) -> usize {
        self.n_seq_max
    }

    /// Clears the batch, i.e. setting its token count to 0.
    pub fn clear(&mut self) {
        tracing::trace!("batch_clear");
        self.data.n_tokens = 0;
    }

    /// Add token to the batch.
    ///
    /// # Panics
    ///
    /// Panics if batch has no token buffer, or the buffer is full, or one of
    /// the `seq_ids` is out of bounds.
    ///
    /// # Safety
    ///
    /// This is unsafe, because it assumes the provided token is valid.
    pub unsafe fn add_token(&mut self, id: Token, pos: Pos, seq_ids: &[SeqId], logits: bool) {
        /* from llama.cpp/common.common.cpp
        ```cpp
        void llama_batch_add(
                        struct llama_batch & batch,
                                llama_token   id,
                                llama_pos   pos,
            const std::vector<llama_seq_id> & seq_ids,
                                    bool   logits) {
            batch.token   [batch.n_tokens] = id;
            batch.pos     [batch.n_tokens] = pos,
            batch.n_seq_id[batch.n_tokens] = seq_ids.size();
            for (size_t i = 0; i < seq_ids.size(); ++i) {
                batch.seq_id[batch.n_tokens][i] = seq_ids[i];
            }
            batch.logits  [batch.n_tokens] = logits;

            batch.n_tokens++;
        }
        ```
        */

        let n_tokens = self.data.n_tokens as usize;
        tracing::trace!(id, pos, logits, seq_ids = ?seq_ids, n_tokens, "batch_add");

        assert_eq!(self.n_embd, 0, "this batch has no token buffer");
        assert!(!self.data.token.is_null()); // this is a bug
        assert!(seq_ids.len() <= self.n_seq_max);
        assert!(n_tokens < self.tokens_buf_size, "batch buffer is full");

        unsafe {
            let token_buf = slice::from_raw_parts_mut(self.data.token, self.tokens_buf_size);
            token_buf[n_tokens] = id;

            let pos_buf = slice::from_raw_parts_mut(self.data.pos, self.tokens_buf_size);
            pos_buf[n_tokens] = pos;

            let n_seq_id_buf = slice::from_raw_parts_mut(self.data.n_seq_id, self.tokens_buf_size);
            n_seq_id_buf[n_tokens] = seq_ids.len() as _;

            for i in 0..seq_ids.len() {
                let seq_id_buf = slice::from_raw_parts_mut(self.data.seq_id, self.tokens_buf_size);
                let seq_id_buf = slice::from_raw_parts_mut(seq_id_buf[n_tokens], self.n_seq_max);
                seq_id_buf[i] = seq_ids[i];
            }

            let logits_buf = slice::from_raw_parts_mut(self.data.logits, self.tokens_buf_size);
            logits_buf[n_tokens] = logits as _;
        }

        self.data.n_tokens += 1;
    }

    /// Sets the `logits` flag for the last element in this batch.
    #[allow(dead_code)]
    pub fn set_last_logits(&mut self, logits: bool) {
        let Some(n_tokens) = (self.data.n_tokens as usize).checked_sub(1)
        else {
            return;
        };
        assert!(n_tokens < self.tokens_buf_size);

        unsafe {
            let logits_buf = slice::from_raw_parts_mut(self.data.logits, n_tokens);
            logits_buf[n_tokens] = logits as i8;
        }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe {
            tracing::trace!("calling llama_batch_free");
            llama_cpp_sys::llama_batch_free(self.data);
        }
    }
}
