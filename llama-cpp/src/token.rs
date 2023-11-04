use crate::backend::model::{
    Model,
    TokenDecoder,
};
pub use crate::backend::{
    Token,
    TokenType,
};

pub trait IntoTokens {
    fn into_tokens(self, model: &Model) -> Vec<Token>;
}

pub trait FromToken: Sized {
    type State: Unpin;

    fn init(model: &Model) -> Self::State;
    fn from_token(token: Token, state: &mut Self::State) -> Option<Self>;
}

impl IntoTokens for Vec<Token> {
    fn into_tokens(self, _model: &Model) -> Vec<Token> {
        self
    }
}

impl FromToken for Token {
    type State = ();

    fn init(_model: &Model) -> Self::State {}

    fn from_token(token: Token, _state: &mut Self::State) -> Option<Self> {
        Some(token)
    }
}

impl IntoTokens for &str {
    fn into_tokens(self, model: &Model) -> Vec<Token> {
        model.tokenize(self, false, false)
    }
}

impl FromToken for String {
    type State = TokenDecoder;

    fn init(model: &Model) -> TokenDecoder {
        model.token_decoder()
    }

    fn from_token(token: Token, token_decoder: &mut TokenDecoder) -> Option<Self> {
        token_decoder.push_token(token);
        token_decoder.pop_string()
    }
}

impl FromToken for Vec<u8> {
    type State = Model;

    fn init(model: &Model) -> Self::State {
        model.clone()
    }

    fn from_token(token: Token, model: &mut Model) -> Option<Self> {
        let mut output = vec![];
        model.token_to_piece(token, &mut output);
        (!output.is_empty()).then_some(output)
    }
}

pub struct BeginOfSequence;

impl IntoTokens for BeginOfSequence {
    fn into_tokens(self, model: &Model) -> Vec<Token> {
        vec![model.token_bos()]
    }
}

pub struct EndOfSequence;

impl IntoTokens for EndOfSequence {
    fn into_tokens(self, model: &Model) -> Vec<Token> {
        vec![model.token_eos()]
    }
}

pub struct Tokenize<'a> {
    pub text: &'a str,
    pub add_bos: bool,
    pub allow_special: bool,
}

impl<'a> IntoTokens for Tokenize<'a> {
    fn into_tokens(self, model: &Model) -> Vec<Token> {
        model.tokenize(self.text, self.add_bos, self.allow_special)
    }
}
