use derivative::Derivative;

use super::{
    context::Context,
    Token,
};

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Grammar {
    #[derivative(Debug = "ignore")]
    pub(super) handle: *mut llama_cpp_sys::llama_grammar,
}

unsafe impl Send for Grammar {}

impl Grammar {
    pub fn accept_token(&mut self, context: &mut Context, token: Token) {
        unsafe {
            llama_cpp_sys::llama_grammar_accept_token(context.handle, self.handle, token);
        }
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

#[derive(Debug, Default)]
pub struct GrammarBuilder {
    rules: Vec<Vec<llama_cpp_sys::llama_grammar_element>>,
    start_rule: usize,
}

impl GrammarBuilder {
    // todo

    pub fn build(self) -> Grammar {
        let handle = unsafe {
            // llama_grammar_init copies the rules into an internal vector, so it's safe to
            // drop our Vec.
            let mut rules = self
                .rules
                .iter()
                .map(|rule| rule.as_ptr())
                .collect::<Vec<_>>();
            llama_cpp_sys::llama_grammar_init(rules.as_mut_ptr(), self.rules.len(), self.start_rule)
        };
        Grammar { handle }
    }
}
