//! Grammar loaded into llama.cpp

use derivative::Derivative;

use super::{
    context::Context,
    Token,
};
use crate::grammar::compiler::Compiled;

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Grammar {
    #[derivative(Debug = "ignore")]
    pub(super) handle: *mut llama_cpp_sys::llama_grammar,
}

unsafe impl Send for Grammar {}

impl Grammar {
    /// # Panics
    ///
    /// Panics if the compiled grammar is invalid.
    pub fn load(compiled: &Compiled) -> Self {
        tracing::trace!("loading grammar: {:#?}", compiled);

        // this checks that the grammar is valid such that it won't cause any UB below.
        compiled.check().expect("grammar invalid");

        let handle = unsafe {
            // llama_grammar_init copies the rules into an internal vector, so it's safe to
            // drop this or `compiled`.
            let mut rules = compiled
                .rules
                .iter()
                .map(|rule| rule.as_ptr())
                .collect::<Vec<_>>();

            llama_cpp_sys::llama_grammar_init(rules.as_mut_ptr(), rules.len(), compiled.root)
        };

        assert!(!handle.is_null());
        Self { handle }
    }

    /// # Safety
    ///
    /// llama.cpp will abort the program, if stacks are empty. stacks can be
    /// empty if we accept tokens that the grammar can't accept.
    /// So it's important to make sure to only accept tokens that were actually
    /// sampled from the grammar.
    pub unsafe fn accept_token(&mut self, context: &mut Context, token: Token) {
        tracing::trace!("accept token: {}", token);
        llama_cpp_sys::llama_grammar_accept_token(context.handle, self.handle, token);
    }
}

impl Drop for Grammar {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys::llama_grammar_free(self.handle);
        }
    }
}

impl Clone for Grammar {
    fn clone(&self) -> Self {
        let handle = unsafe { llama_cpp_sys::llama_grammar_copy(self.handle) };
        Self { handle }
    }
}
