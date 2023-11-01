//! Compile from AST to binary format.

use std::collections::HashMap;

use super::{
    ast::{
        Alternatives,
        CharClass,
        CharRange,
        Grammar,
        Id,
        Literal,
        Qualified,
        Qualifier,
        Sequence,
        Term,
    },
    Error,
};
use crate::{
    backend::grammar::{
        Compiled,
        Element,
        ElementType,
    },
    utils::IsLast,
};

enum Rule<'source> {
    Placeholder(Option<Id<'source>>),
    Ready(Vec<Element>),
}

impl<'source> Rule<'source> {
    fn get_ready(self) -> Vec<Element> {
        match self {
            // if our compiler works correctly, there should be no placeholders left
            Rule::Placeholder(id) => panic!("placeholder left in rules: id={id:?}"),
            Rule::Ready(rule) => rule,
        }
    }
}

/// Compiles an AST ([`ast`](super::ast)) into a binary grammar ([`Compiled`]).
#[derive(Default)]
pub struct Compiler<'source> {
    names: HashMap<Id<'source>, usize>,
    rules: Vec<Rule<'source>>,
}

impl<'source> Compiler<'source> {
    /// Compile into binary format.
    ///
    /// # Panics
    ///
    /// Panics if the compiled grammar is invalid. This indicates a bug in the
    /// compiler.
    pub fn finish(self, root: Id<'source>) -> Result<Compiled, Error> {
        let root = self.get_rule_index(root).map_err(|_| Error::NoRoot)?;

        let rules = self
            .rules
            .into_iter()
            .map(|rule| rule.get_ready())
            .collect();

        let compiled = Compiled { root, rules };

        compiled
            .check()
            .expect("compiled grammar is invalid. this is a bug.");

        Ok(compiled)
    }

    /// Push an [`ast::Grammar`](Grammar) node into the compiler.
    pub fn push_ast<'ast>(&mut self, grammar: &'ast Grammar<'source>) -> Result<(), Error> {
        grammar.compile(self, ())?;
        Ok(())
    }

    /// Create a new rule without code (yet) and return its index.
    pub fn new_placeholder(&mut self, id: Option<Id<'source>>) -> usize {
        let index = self.rules.len();
        self.rules.push(Rule::Placeholder(id));
        if let Some(id) = id {
            self.names.insert(id, index);
        }
        index
    }

    /// Create a new rule from binary format and return its index
    pub fn new_ready(&mut self, buffer: Buffer) -> usize {
        let index = self.rules.len();
        self.rules.push(Rule::Ready(buffer.elements));
        index
    }

    /// Finalize a rule placeholder with its binary format
    ///
    /// # Panics
    ///
    /// Panics if the rule index is invalid, or the rule is not a placeholder.
    pub fn finish_rule(&mut self, index: usize, buffer: Buffer) {
        let rule = self
            .rules
            .get_mut(index)
            .unwrap_or_else(|| panic!("invalid rule index: {index}"));
        assert!(matches!(rule, Rule::Placeholder(_)));
        *rule = Rule::Ready(buffer.elements);
    }

    /// Get rule index by its symbol.
    pub fn get_rule_index(&self, id: Id<'source>) -> Result<usize, Error> {
        self.names
            .get(&id)
            .ok_or_else(|| {
                Error::UndefinedSymbol {
                    symbol: id.to_string(),
                }
            })
            .copied()
    }
}

/// Rule buffer that can be used to construct individual rules in their binary
/// format.
#[derive(Debug, Default)]
pub struct Buffer {
    pub elements: Vec<Element>,
}

impl Buffer {
    fn push(&mut self, ty: ElementType, value: u32) {
        self.elements.push(Element { type_: ty, value })
    }

    pub fn end(&mut self) {
        self.push(ElementType::LLAMA_GRETYPE_END, 0);
    }

    pub fn alt(&mut self) {
        self.push(ElementType::LLAMA_GRETYPE_ALT, 0)
    }

    pub fn rule_ref(&mut self, index: usize) {
        self.push(ElementType::LLAMA_GRETYPE_RULE_REF, index as _);
    }

    pub fn char(&mut self, c: char) {
        self.push(ElementType::LLAMA_GRETYPE_CHAR, c as _);
    }

    pub fn char_not(&mut self, c: char) {
        self.push(ElementType::LLAMA_GRETYPE_CHAR_NOT, c as _);
    }

    pub fn char_range(&mut self, c: char) {
        self.push(ElementType::LLAMA_GRETYPE_CHAR_RNG_UPPER, c as _);
    }

    pub fn char_alt(&mut self, c: char) {
        self.push(ElementType::LLAMA_GRETYPE_CHAR_ALT, c as _);
    }
}

trait Compile<'source> {
    type Context<'context>: 'context
    where
        Self: 'context;

    fn compile<'ast, 'context>(
        &'ast self,
        compiler: &mut Compiler<'source>,
        context: Self::Context<'context>,
    ) -> Result<(), Error>;
}

impl<'source> Compile<'source> for Grammar<'source> {
    type Context<'context> = () where Self: 'context;

    fn compile<'ast, 'context>(
        &'ast self,
        compiler: &mut Compiler<'source>,
        _: (),
    ) -> Result<(), Error> {
        for production in &self.0 {
            if compiler.names.contains_key(&production.lhs) {
                return Err(Error::DuplicateSymbol {
                    symbol: production.lhs.to_string(),
                });
            }
            compiler.new_placeholder(Some(production.lhs));
        }

        for production in &self.0 {
            let mut buffer = Buffer::default();
            production.rhs.compile(compiler, &mut buffer)?;
            buffer.end();

            let index = compiler.get_rule_index(production.lhs).unwrap();
            compiler.finish_rule(index, buffer);
        }

        Ok(())
    }
}

impl<'source> Compile<'source> for Alternatives<'source> {
    type Context<'context> = &'context mut Buffer where Self: 'context;

    fn compile<'ast>(
        &'ast self,
        compiler: &mut Compiler<'source>,
        buffer: &mut Buffer,
    ) -> Result<(), Error> {
        for (sequence, is_last) in IsLast::new(self.0.iter()) {
            sequence.compile(compiler, buffer)?;
            if !is_last {
                buffer.alt();
            }
        }

        Ok(())
    }
}

impl<'source> Compile<'source> for Sequence<'source> {
    type Context<'context> = &'context mut Buffer where Self: 'context;

    fn compile<'ast, 'context>(
        &'ast self,
        compiler: &mut Compiler<'source>,
        buffer: &'context mut Buffer,
    ) -> Result<(), Error> {
        for qualified in &self.0 {
            qualified.compile(compiler, buffer)?;
        }

        Ok(())
    }
}

impl<'source> Compile<'source> for Qualified<'source> {
    type Context<'context> = &'context mut Buffer where Self: 'context;

    fn compile<'ast, 'context>(
        &'ast self,
        compiler: &mut Compiler<'source>,
        buffer: &'context mut Buffer,
    ) -> Result<(), Error> {
        match self.qualifier {
            Some(qualifier) => {
                // buffer for the generated rule
                let mut buffer_gen = Buffer::default();
                let index_gen = compiler.new_placeholder(None);

                match qualifier {
                    Qualifier::Optional => {
                        // S? --> S' ::= S |
                        self.term.compile(compiler, &mut buffer_gen)?;
                        buffer_gen.alt();
                        buffer_gen.end();
                    }
                    Qualifier::Many0 => {
                        // S* --> S' ::= S S' |
                        self.term.compile(compiler, &mut buffer_gen)?;
                        buffer_gen.rule_ref(index_gen);
                        buffer_gen.alt();
                        buffer_gen.end();
                    }
                    Qualifier::Many1 => {
                        // compile sub expression into separate rule
                        let mut buffer_sub = Buffer::default();
                        self.term.compile(compiler, &mut buffer_sub)?;
                        buffer_sub.end();
                        let index_sub = compiler.new_ready(buffer_sub);

                        // S+ --> S' ::= S S' | S
                        buffer_gen.rule_ref(index_sub);
                        buffer_gen.rule_ref(index_gen);
                        buffer_gen.alt();
                        buffer_gen.rule_ref(index_sub);
                        buffer_gen.end();
                    }
                }

                compiler.finish_rule(index_gen, buffer_gen);
                buffer.rule_ref(index_gen);
            }
            None => {
                self.term.compile(compiler, buffer)?;
            }
        }

        Ok(())
    }
}

impl<'source> Compile<'source> for Term<'source> {
    type Context<'context> = &'context mut Buffer where Self: 'context;

    fn compile<'ast, 'context>(
        &'ast self,
        compiler: &mut Compiler<'source>,
        buffer: &'context mut Buffer,
    ) -> Result<(), Error> {
        match self {
            Term::Literal(literal) => literal.compile(compiler, buffer)?,
            Term::CharClass(char_class) => char_class.compile(compiler, buffer)?,
            Term::Parenthesis(alternatives) => {
                let mut buffer_sub = Buffer::default();
                alternatives.compile(compiler, &mut buffer_sub)?;
                buffer_sub.end();
                let index_sub = compiler.new_ready(buffer_sub);
                buffer.rule_ref(index_sub);
            }
            Term::Rule(id) => {
                let index = compiler.get_rule_index(*id)?;
                buffer.rule_ref(index);
            }
        }

        Ok(())
    }
}

impl<'source> Compile<'source> for Literal {
    type Context<'context> = &'context mut Buffer where Self: 'context;

    fn compile<'ast, 'context>(
        &'ast self,
        _compiler: &mut Compiler<'source>,
        buffer: &'context mut Buffer,
    ) -> Result<(), Error> {
        for c in self.0.chars() {
            buffer.char(c);
        }

        Ok(())
    }
}

impl<'source> Compile<'source> for CharClass {
    type Context<'context> = &'context mut Buffer where Self: 'context;

    fn compile<'ast, 'context>(
        &'ast self,
        _compiler: &mut Compiler<'source>,
        buffer: &'context mut Buffer,
    ) -> Result<(), Error> {
        let mut char_ranges = self.char_ranges.iter();

        enum CharType {
            Start,
            StartNegated,
            Alt,
        }

        let mut char_range = |start, end, ty| {
            match ty {
                CharType::Start => buffer.char(start),
                CharType::StartNegated => buffer.char_not(start),
                CharType::Alt => buffer.char_alt(start),
            }
            match end {
                // valid range
                Some(end) if start < end => buffer.char_range(end),
                // invalid range
                Some(end) if start > end => return Err(Error::InvalidCharRange { start, end }),
                // either start == end or end is None
                _ => {}
            }
            Ok(())
        };

        if let Some(first) = char_ranges.next() {
            if self.negated {
                match first {
                    CharRange::Single(c) => char_range(*c, None, CharType::StartNegated)?,
                    CharRange::Range { start, end } => {
                        char_range(*start, Some(*end), CharType::StartNegated)?
                    }
                }
            }
            else {
                match first {
                    CharRange::Single(c) => char_range(*c, None, CharType::Start)?,
                    CharRange::Range { start, end } => {
                        char_range(*start, Some(*end), CharType::Start)?
                    }
                }
            }

            while let Some(next) = char_ranges.next() {
                match next {
                    CharRange::Single(c) => char_range(*c, None, CharType::Alt)?,
                    CharRange::Range { start, end } => {
                        char_range(*start, Some(*end), CharType::Alt)?
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::grammar::{
        parse_and_compile,
        Error,
    };

    #[test]
    fn it_compiles_chess_example() {
        let s = r#"
# Specifies chess moves as a list in algebraic notation, using PGN conventions

# Force first move to "1. ", then any 1-2 digit number after, relying on model to follow the pattern
root    ::= "1. " move " " move "\n" ([1-9] [0-9]? ". " move " " move "\n")+
move    ::= (pawn | nonpawn | castle) [+#]?

# piece type, optional file/rank, optional capture, dest file & rank
nonpawn ::= [NBKQR] [a-h]? [1-8]? "x"? [a-h] [1-8]

# optional file & capture, dest file & rank, optional promotion
pawn    ::= ([a-h] "x")? [a-h] [1-8] ("=" [NBKQR])?

castle  ::= "O-O" "-O"?
        "#;

        let compiled = parse_and_compile(s).unwrap();
        compiled.check().unwrap();
    }

    #[test]
    fn it_doesnt_compile_duplicate_symbols() {
        let s = r#"
root ::= foo*
foo ::= "foo"
foo ::= "bar"
        "#;

        match parse_and_compile(s) {
            Err(Error::DuplicateSymbol { symbol }) => assert_eq!(symbol, "foo"),
            _ => panic!("expected Error::DuplicateSymbol"),
        }
    }

    #[test]
    fn it_doesnt_compile_without_root() {
        let s = r#"
foo ::= "foo"
        "#;

        match parse_and_compile(s) {
            Err(Error::NoRoot) => {}
            Err(e) => panic!("expected Error::NoRoot but got another error: {e}"),
            _ => panic!("expected Error::NoRootm but got Ok(_)"),
        }
    }

    #[test]
    fn it_doesnt_compile_undefined_ref() {
        let s = r#"
root ::= foo
bar ::= "bar"
        "#;

        match parse_and_compile(s) {
            Err(Error::UndefinedSymbol { symbol }) => assert_eq!(symbol, "foo"),
            _ => panic!("expected Error::UndefinedSymbol"),
        }
    }
}
