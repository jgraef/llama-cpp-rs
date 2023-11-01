//! LLM inference

use itertools::Itertools;

use super::{
    batch::Batch,
    context::{
        Context,
        DecodeWarning,
    },
    sampling::Sampler,
    Error,
    Pos,
    Token,
};
use crate::{
    backend::sampling::Candidates,
    utils::IsLast,
};

/// Helper for batched decoding and sampling.
pub struct Inference<'a> {
    context: &'a mut Context,
    batch: Batch,
    pos: Pos,
    logits_pos: Option<i32>,
}

impl<'a> Inference<'a> {
    /// # Panics
    /// 
    /// Panics if the `batch_size` is 0.
    pub fn new(context: &'a mut Context, batch_size: usize) -> Self {
        assert!(batch_size > 0);

        // for now we'll have `n_seq_max=1`.
        let batch = Batch::new(batch_size, 1);

        Self {
            context,
            batch,
            pos: 0,
            logits_pos: None,
        }
    }

    /// Push tokens for decoding.
    pub fn push(&mut self, tokens: &[Token]) -> Result<(), Error> {
        for chunk in &IsLast::new(tokens.into_iter()).chunks(self.batch.size()) {
            self.batch.clear();

            for (token, is_last) in chunk {
                // add token to batch
                self.batch.add(*token, self.pos, &[0], is_last);

                // generate logits for the last position
                if is_last {
                    self.logits_pos = Some(self.pos);
                }

                self.pos += 1;
            }

            // decode
            if let Some(warning) = self.context.decode(&self.batch)? {
                // todo
                tracing::warn!("{}", warning);
                //warnings.push(warning);
            }
        }

        Ok(())
    }

    /// Samples new tokens and decodes them.
    /// 
    /// # Panics
    /// 
    /// Panics if no tokens have been pushed for decoding.
    /// 
    pub fn sample(&mut self, sampler: &mut Sampler) -> Result<SampleResult, Error> {
        // todo: this should return an error, right?
        let logits_pos = self
            .logits_pos
            .expect("sample was called before a prompt was feed into inference");
        let logits = self.context.get_logits_ith(logits_pos);

        // clear batch to feed in sampled token
        self.batch.clear();

        // create candidates from logits
        let mut candidates = Candidates::new(logits);

        // sample next token
        let next_token = sampler.sample(&mut candidates, &mut self.context);

        // add sampled token to batch
        self.batch.add(next_token, self.pos, &[0], true);

        // generate logits for the only token that is in the batch
        self.logits_pos = Some(0);

        self.pos += 1;

        // decode
        let warning = self.context.decode(&mut self.batch)?;

        Ok(SampleResult {
            token: next_token,
            warning,
        })
    }

    pub fn get_embeddings(&self) -> &[f32] {
        self.context.get_embeddings()
    }
}


#[derive(Copy, Clone, Debug)]
pub struct SampleResult {
    pub token: Token,
    pub warning: Option<DecodeWarning>,
}
