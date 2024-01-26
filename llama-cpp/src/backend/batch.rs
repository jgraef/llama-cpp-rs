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
    n_embd: u32,
    n_tokens_max: u32,
    n_seq_max: u32,
}

unsafe impl Send for Batch {}

impl Batch {
    /// Create new batch that will accept tokens
    ///
    /// # Panics
    ///
    /// Panics if `n_tokens` is 0.
    pub fn new_tokens(n_tokens_max: u32, n_seq_max: u32) -> Self {
        Self::new(n_tokens_max, 0, n_seq_max)
    }

    fn new(n_tokens_max: u32, n_embd: u32, n_seq_max: u32) -> Self {
        assert!(n_tokens_max > 0);
        assert!(n_seq_max > 0);

        tracing::trace!(n_tokens_max, n_seq_max, "calling llama_batch_init");
        let data = unsafe {
            // for `embd` we need to pass 0, otherwise data.tokens is not allocated and thus
            // we are unsafe. we should write a separate `Batch` for embedding.
            llama_cpp_sys::llama_batch_init(n_tokens_max as _, n_embd as _, n_seq_max as _)
        };

        Self {
            data,
            n_embd,
            n_tokens_max,
            n_seq_max,
        }
    }

    /// Returns the size of the token buffer.
    pub fn n_tokens_max(&self) -> u32 {
        self.n_tokens_max
    }

    /// Returns the embedding size, or 0 if this buffer can't accept embeddings.
    pub fn n_embd(&self) -> u32 {
        self.n_embd
    }

    /// Returns the max number of sequences.
    pub fn n_seq_max(&self) -> u32 {
        self.n_seq_max
    }

    /// Clears the batch, i.e. setting its token count to 0.
    pub fn clear(&mut self) {
        tracing::trace!("batch_clear");
        self.data.n_tokens = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.data.n_tokens == 0
    }

    pub fn is_full(&self) -> bool {
        self.data.n_tokens as u32 == self.n_tokens_max
    }

    pub fn n_tokens(&self) -> u32 {
        self.data.n_tokens as u32
    }

    /// Add token to the batch.
    ///
    /// # Panics
    ///
    /// Panics if batch has no token buffer, or the buffer is full, or one of
    /// the `seq_ids` is out of bounds.
    pub fn add_token(&mut self, id: Token, pos: Pos, seq_ids: &[SeqId], logits: bool) {
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
        let n_tokens_max = self.n_tokens_max as usize;

        tracing::trace!(id, pos, logits, seq_ids = ?seq_ids, n_tokens, n_tokens_max, "batch_add");

        assert_eq!(self.n_embd, 0, "this batch has no token buffer");
        assert!(!self.data.token.is_null()); // this is a bug
        assert!(seq_ids.len() <= self.n_seq_max as _);
        assert!(n_tokens < n_tokens_max, "batch buffer is full");

        unsafe {
            let token_buf = slice::from_raw_parts_mut(self.data.token, n_tokens_max);
            token_buf[n_tokens] = id;

            let pos_buf = slice::from_raw_parts_mut(self.data.pos, n_tokens_max);
            pos_buf[n_tokens] = pos;

            let n_seq_id_buf = slice::from_raw_parts_mut(self.data.n_seq_id, n_tokens_max);
            n_seq_id_buf[n_tokens] = seq_ids.len() as _;

            let seq_id_buf = slice::from_raw_parts_mut(self.data.seq_id, n_tokens_max);
            for i in 0..seq_ids.len() {
                let seq_id_buf =
                    slice::from_raw_parts_mut(seq_id_buf[n_tokens], self.n_seq_max as _);
                seq_id_buf[i] = seq_ids[i];
            }

            let logits_buf = slice::from_raw_parts_mut(self.data.logits, n_tokens_max);
            logits_buf[n_tokens] = logits as _;
        }

        self.data.n_tokens += 1;
    }

    pub fn resize(&mut self, n_tokens_max: u32, n_seq_max: u32) {
        if self.n_embd != 0 {
            todo!("resize embedding buffer");
        }

        unsafe {
            let new_batch = llama_cpp_sys::llama_batch_init(
                n_tokens_max as _,
                self.n_embd as _,
                n_seq_max as _,
            );

            let n = std::cmp::min(self.data.n_tokens as usize, n_tokens_max as usize);

            let old_token_buf = slice::from_raw_parts(self.data.token, n);
            let old_pos_buf = slice::from_raw_parts(self.data.pos, n);
            let old_n_seq_id_buf = slice::from_raw_parts(self.data.n_seq_id, n);
            let old_logits_buf = slice::from_raw_parts(self.data.logits, n);

            let new_token_buf = slice::from_raw_parts_mut(new_batch.token, n);
            let new_pos_buf = slice::from_raw_parts_mut(new_batch.pos, n);
            let new_n_seq_id_buf = slice::from_raw_parts_mut(new_batch.n_seq_id, n);
            let new_logits_buf = slice::from_raw_parts_mut(new_batch.logits, n);

            new_token_buf.copy_from_slice(old_token_buf);
            new_pos_buf.copy_from_slice(old_pos_buf);
            new_n_seq_id_buf.copy_from_slice(old_n_seq_id_buf);
            new_logits_buf.copy_from_slice(old_logits_buf);

            let old_seq_id_buf = slice::from_raw_parts(self.data.seq_id, n);
            let new_seq_id_buf = slice::from_raw_parts_mut(new_batch.seq_id, n);

            for i in 0..n {
                let m = old_n_seq_id_buf[i] as usize;
                let old_seq_id_buf = slice::from_raw_parts(old_seq_id_buf[i], m);
                let new_seq_id_buf = slice::from_raw_parts_mut(new_seq_id_buf[i], m);
                new_seq_id_buf.copy_from_slice(old_seq_id_buf);
            }

            self.n_tokens_max = n_tokens_max;
            self.n_seq_max = n_seq_max;
            self.data = new_batch;
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
