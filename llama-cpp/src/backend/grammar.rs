//! Grammar loaded into llama.cpp

use derivative::Derivative;

use super::{
    context::Context,
    Token,
};
use crate::utils::IsLast;

/// Alias for llama.cpp's grammar element.
pub type Element = llama_cpp_sys::llama_grammar_element;

/// Alias for llama.cpp's grammar element type.
pub type ElementType = llama_cpp_sys::llama_gretype;

/// Grammar that has been loaded into llama.cpp and is ready for sampling.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Loaded {
    #[derivative(Debug = "ignore")]
    pub(super) handle: *mut llama_cpp_sys::llama_grammar,
}

unsafe impl Send for Loaded {}

impl Loaded {
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

impl Drop for Loaded {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys::llama_grammar_free(self.handle);
        }
    }
}

impl Clone for Loaded {
    fn clone(&self) -> Self {
        let handle = unsafe { llama_cpp_sys::llama_grammar_copy(self.handle) };
        Self { handle }
    }
}

/// The compiled grammar is invalid.
#[derive(Debug, thiserror::Error)]
pub enum CheckError {
    #[error("no rules")]
    NoRules,

    #[error("no root: index={index}")]
    InvalidRoot { index: usize },

    #[error("empty rule: rule={rule}")]
    EmptyRule { rule: usize },

    #[error("no end in rule: rule={rule}, pos={pos}")]
    NoEnd { rule: usize, pos: usize },

    #[error("end in middle of rule: rule={rule}, pos={pos}")]
    EndInMiddle { rule: usize, pos: usize },

    #[error("invalid rule ref: rule={rule}, pos={pos}, ref={rule_ref}")]
    InvalidRef {
        rule: usize,
        pos: usize,
        rule_ref: u32,
    },

    #[error("invalid code point: rule={rule}, pos={pos}, code_point={code_point}")]
    InvalidChar {
        rule: usize,
        pos: usize,
        code_point: u32,
    },
}

/// A compiled grammar. This is the binary format that llama.cpp uses.
#[derive(Clone, Debug)]
pub struct Compiled {
    /// Index of the root rule.
    pub root: usize,

    /// The grammar rules. Each rule is a `Vec` of [`Element`]s.
    pub rules: Vec<Vec<Element>>,
}

impl Compiled {
    /// Checks if the compiled grammar is valid.
    pub fn check(&self) -> Result<(), CheckError> {
        // todo: i think we really need to make sure the grammar is correct, otherwise
        // loading and running it, could lead to UB.
        let n_rules = self.rules.len();

        if n_rules == 0 {
            return Err(CheckError::NoRules);
        }

        if self.root >= n_rules {
            return Err(CheckError::InvalidRoot { index: self.root });
        }

        for (i, rule) in self.rules.iter().enumerate() {
            if rule.is_empty() {
                return Err(CheckError::EmptyRule { rule: i });
            }

            for ((pos, element), is_last) in IsLast::new(rule.iter().enumerate()) {
                if is_last && !matches!(element.type_, ElementType::LLAMA_GRETYPE_END) {
                    return Err(CheckError::NoEnd { rule: i, pos });
                }
                match element.type_ {
                    ElementType::LLAMA_GRETYPE_END if !is_last => {
                        return Err(CheckError::EndInMiddle { rule: i, pos });
                    }
                    ElementType::LLAMA_GRETYPE_RULE_REF if element.value as usize >= n_rules => {
                        return Err(CheckError::InvalidRef {
                            rule: i,
                            pos,
                            rule_ref: element.value,
                        });
                    }
                    ElementType::LLAMA_GRETYPE_CHAR
                    | ElementType::LLAMA_GRETYPE_CHAR_NOT
                    | ElementType::LLAMA_GRETYPE_CHAR_RNG_UPPER
                    | ElementType::LLAMA_GRETYPE_CHAR_ALT => {
                        if let Err(_e) = char::try_from(element.value) {
                            return Err(CheckError::InvalidChar {
                                rule: i,
                                pos,
                                code_point: element.value,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    /// # Panics
    ///
    /// Panics if the compiled grammar is invalid.
    pub fn load(&self) -> Loaded {
        tracing::trace!("loading grammar: {:#?}", self);

        // this checks that the grammar is valid such that it won't cause any UB below.
        self.check().expect("grammar invalid");

        let handle = unsafe {
            // llama_grammar_init copies the rules into an internal vector, so it's safe to
            // drop this or `compiled`.
            let mut rules = self
                .rules
                .iter()
                .map(|rule| rule.as_ptr())
                .collect::<Vec<_>>();

            llama_cpp_sys::llama_grammar_init(rules.as_mut_ptr(), rules.len(), self.root)
        };

        assert!(!handle.is_null());
        Loaded { handle }
    }
}
