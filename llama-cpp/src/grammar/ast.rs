#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grammar<'source>(pub Vec<Production<'source>>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Production<'source> {
    pub lhs: Id<'source>,
    pub rhs: Alternatives<'source>,
}

#[derive(
    Copy,
    Clone,
    Debug,
    Hash,
    PartialEq,
    Eq,
    derive_more::Display,
    derive_more::From,
    derive_more::AsRef,
)]
pub struct Id<'source>(pub(super) &'source str);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Alternatives<'source>(pub Vec<Sequence<'source>>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sequence<'source>(pub Vec<Qualified<'source>>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qualified<'source> {
    pub term: Term<'source>,
    pub qualifier: Option<Qualifier>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Qualifier {
    Optional,
    Many0,
    Many1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Term<'source> {
    Literal(Literal),
    CharClass(CharClass),
    Parenthesis(Box<Alternatives<'source>>),
    Rule(Id<'source>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Literal(pub String);

impl From<String> for Literal {
    fn from(value: String) -> Self {
        Literal(value)
    }
}

impl<'a> From<&'a str> for Literal {
    fn from(value: &'a str) -> Self {
        value.to_owned().into()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CharClass {
    pub char_ranges: Vec<CharRange>,
    pub negated: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CharRange {
    Single(char),
    Range { start: char, end: char },
}
