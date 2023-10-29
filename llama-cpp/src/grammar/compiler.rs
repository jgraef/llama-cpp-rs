use std::collections::HashMap;

use super::{
    ast::{
        Alternatives,
        CharClass,
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
    backend::grammar::Grammar as BackendGrammar,
    grammar::ast::CharRange,
    utils::IsLast,
};

pub type Element = llama_cpp_sys::llama_grammar_element;
pub type ElementType = llama_cpp_sys::llama_gretype;

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

#[derive(Clone, Debug)]
pub struct Compiled {
    pub root: usize,
    pub rules: Vec<Vec<Element>>,
}

impl Compiled {
    pub fn load(&self) -> BackendGrammar {
        // the BackendGrammar::load calls check on this.
        BackendGrammar::load(self)
    }

    pub fn check(&self) -> Result<(), Error> {
        // todo: i think we really need to make sure the grammar is correct, otherwise
        // loading and running it, could lead to UB.
        let n_rules = self.rules.len();

        if n_rules == 0 {
            return Err(Error::InvalidCompiled("no rules"));
        }

        if self.root >= n_rules {
            return Err(Error::InvalidCompiled("no root"));
        }

        for rule in &self.rules {
            if rule.is_empty() {
                return Err(Error::InvalidCompiled("empty rule"));
            }

            for (element, is_last) in IsLast::new(rule.iter()) {
                if is_last && !matches!(element.type_, ElementType::LLAMA_GRETYPE_END) {
                    return Err(Error::InvalidCompiled("no end"));
                }
                match element.type_ {
                    ElementType::LLAMA_GRETYPE_END if !is_last => {
                        return Err(Error::InvalidCompiled("end in the middle of a rule"));
                    }
                    ElementType::LLAMA_GRETYPE_RULE_REF => {
                        if element.value as usize >= n_rules {
                            return Err(Error::InvalidCompiled("ref index out of bounds"));
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }
}

#[derive(Default)]
pub(super) struct Compiler<'source> {
    names: HashMap<Id<'source>, usize>,
    rules: Vec<Rule<'source>>,
}

impl<'source> Compiler<'source> {
    pub fn finish(self, root: Id<'source>) -> Result<Compiled, Error> {
        let root = self.get_rule_index(root)?;

        let rules = self
            .rules
            .into_iter()
            .map(|rule| rule.get_ready())
            .collect();

        Ok(Compiled { root, rules })
    }

    pub fn push_ast<'ast>(&mut self, grammar: &'ast Grammar<'source>) -> Result<(), Error> {
        grammar.compile(self, ())?;
        Ok(())
    }

    fn new_placeholder(&mut self, id: Option<Id<'source>>) -> usize {
        let index = self.rules.len();
        self.rules.push(Rule::Placeholder(id));
        if let Some(id) = id {
            self.names.insert(id, index);
        }
        index
    }

    fn new_ready(&mut self, buffer: Buffer) -> usize {
        let index = self.rules.len();
        self.rules.push(Rule::Ready(buffer.elements));
        index
    }

    fn finish_rule(&mut self, index: usize, buffer: Buffer) {
        let rule = self
            .rules
            .get_mut(index)
            .unwrap_or_else(|| panic!("invalid rule index: {index}"));
        assert!(matches!(rule, Rule::Placeholder(_)));
        *rule = Rule::Ready(buffer.elements);
    }

    fn get_rule_index(&self, id: Id<'source>) -> Result<usize, Error> {
        self.names
            .get(&id)
            .ok_or_else(|| Error::Undefined(id.to_string()))
            .copied()
    }
}

#[derive(Debug, Default)]
struct Buffer {
    elements: Vec<Element>,
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

        if let Some(first) = char_ranges.next() {
            if self.negated {
                match first {
                    CharRange::Single(c) => buffer.char_not(*c),
                    CharRange::Range { start, end } => {
                        buffer.char_not(*start);
                        buffer.char_range(*end);
                    }
                }
            }
            else {
                match first {
                    CharRange::Single(c) => buffer.char(*c),
                    CharRange::Range { start, end } => {
                        buffer.char(*start);
                        buffer.char_range(*end);
                    }
                }
            }

            while let Some(next) = char_ranges.next() {
                match next {
                    CharRange::Single(c) => buffer.char_alt(*c),
                    CharRange::Range { start, end } => {
                        buffer.char_alt(*start);
                        buffer.char_range(*end);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
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

        let ast = crate::grammar::parse(s).unwrap();
        let compiled = crate::grammar::compile(&ast).unwrap();
        println!("{:#?}", compiled);
        compiled.check().unwrap();
    }
}
