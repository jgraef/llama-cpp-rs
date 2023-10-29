use nom::{
    branch::alt,
    bytes::complete::{
        escaped_transform,
        is_not,
        tag,
        take,
        take_while1,
    },
    character::complete::{
        char,
        multispace0,
        multispace1,
        newline,
        none_of,
        one_of,
        space0,
        space1,
    },
    combinator::{
        all_consuming,
        cut,
        map,
        not,
        opt,
        peek,
        value,
    },
    error::{
        context,
        ErrorKind,
        FromExternalError,
        ParseError,
        VerboseError,
    },
    multi::{
        many0,
        many0_count,
        separated_list0,
        separated_list1,
    },
    sequence::{
        delimited,
        pair,
        preceded,
        terminated,
        tuple,
    },
    IResult,
    Parser,
};

use super::ast::{
    Alternatives,
    CharClass,
    CharRange,
    Grammar,
    Id,
    Literal,
    Production,
    Qualified,
    Qualifier,
    Sequence,
    Term,
};

type Res<'a, U> = IResult<&'a str, U, VerboseError<&'a str>>;

/// consumes a single comment
fn consume_comment(input: &str) -> Res<()> {
    value((), pair(char('#'), is_not("\r\n")))(input)
}

trait Ws {
    fn consume<'a>(input: &'a str) -> Res<'a, ()>;
    fn consume1<'a>(input: &'a str) -> Res<'a, ()>;
}

pub struct SingleLine;

impl Ws for SingleLine {
    fn consume<'a>(input: &'a str) -> Res<'a, ()> {
        value((), space0)(input)
    }

    fn consume1<'a>(input: &'a str) -> Res<'a, ()> {
        value((), space1)(input)
    }
}

pub struct MultiLine;

impl Ws for MultiLine {
    fn consume<'a>(input: &'a str) -> Res<'a, ()> {
        value((), multispace0)(input)
    }

    fn consume1<'a>(input: &'a str) -> Res<'a, ()> {
        value((), multispace1)(input)
    }
}

/// consumes whitespace and comments
fn consume_wsc<'a>(input: &'a str) -> Res<'a, ()> {
    value(
        (),
        terminated(
            many0_count(preceded(multispace0, consume_comment)),
            multispace0,
        ),
    )(input)
}

/// consumes whitespace and comments, but at least one whitespace
fn consume1_wsc<'a>(input: &'a str) -> Res<'a, ()> {
    value(
        (),
        terminated(
            many0_count(preceded(multispace0, consume_comment)),
            multispace1,
        ),
    )(input)
}

/// consumes all whitespace and comments before calling the parser `f`
fn wsc<'a, U>(f: impl FnMut(&'a str) -> Res<'a, U>) -> impl FnMut(&'a str) -> Res<'a, U> {
    preceded(consume_wsc, f)
}

pub(super) fn parse_grammar_complete(input: &str) -> Res<Grammar> {
    all_consuming(terminated(parse_grammar, consume_wsc))(input)
}

fn parse_grammar(input: &str) -> Res<Grammar> {
    context(
        "grammar",
        map(
            separated_list0(newline, parse_production),
            //many0(parse_production),
            Grammar,
        ),
    )(input)
}

fn parse_production<'a>(input: &'a str) -> Res<'a, Production> {
    context(
        "production",
        map(
            tuple((parse_id, wsc(tag("::=")), parse_alternatives)),
            |(lhs, _, rhs)| Production { lhs, rhs },
        ),
    )(input)
}

fn parse_id<'a>(input: &'a str) -> Res<'a, Id> {
    context(
        "id",
        map(
            wsc(take_while1(|c: char| c.is_alphanumeric() || c == '-')),
            |id| Id(id),
        ),
    )(input)
}

fn parse_alternatives<'a>(input: &'a str) -> Res<'a, Alternatives> {
    context(
        "alternatives",
        map(
            separated_list1(wsc(char('|')), parse_sequence),
            Alternatives,
        ),
    )(input)
}

fn parse_sequence<'a>(input: &'a str) -> Res<'a, Sequence> {
    context(
        "sequence",
        map(separated_list1(consume1_wsc, parse_qualified), Sequence),
    )(input)
}

fn parse_qualified<'a>(input: &'a str) -> Res<'a, Qualified> {
    context(
        "qualified",
        map(
            tuple((parse_term, opt(parse_qualifier))),
            |(term, qualifier)| Qualified { term, qualifier },
        ),
    )(input)
}

fn parse_qualifier<'a>(input: &'a str) -> Res<'a, Qualifier> {
    context(
        "qualifier",
        map(wsc(one_of("?*+")), |q| {
            match q {
                '?' => Qualifier::Optional,
                '*' => Qualifier::Many0,
                '+' => Qualifier::Many1,
                _ => unreachable!(),
            }
        }),
    )(input)
}

fn parse_term<'a>(input: &'a str) -> Res<'a, Term> {
    context(
        "term",
        alt((
            map(parse_literal, Term::Literal),
            map(parse_char_class, Term::CharClass),
            map(parse_parenthesis, |expression| {
                Term::Parenthesis(Box::new(expression))
            }),
            map(parse_rule_ref, Term::Rule),
        )),
    )(input)
}

fn parse_rule_ref(input: &str) -> Res<'_, Id> {
    terminated(parse_id, peek(not(wsc(tag("::=")))))(input)
}

fn parse_literal<'a>(input: &'a str) -> Res<'a, Literal> {
    context(
        "literal",
        wsc(map(
            delimited(
                char('\"'),
                escaped_transform(none_of("\"\r\n\\"), '\\', parse_literal_escape),
                char('\"'),
            ),
            Literal,
        )),
    )(input)
}

fn parse_escaped_unicode(input: &str) -> Res<char> {
    let (input, code_point) = alt((
        preceded(char('x'), take(2usize).and_then(hex_u32)),
        preceded(char('u'), take(4usize).and_then(hex_u32)),
        preceded(char('U'), take(8usize).and_then(hex_u32)),
    ))(input)?;
    let code_point = char::from_u32(code_point).ok_or_else(|| {
        nom::Err::Error(VerboseError::from_error_kind(
            input,
            ErrorKind::EscapedTransform,
        ))
    })?;
    Ok((input, code_point))
}

fn hex_u32(input: &str) -> Res<u32> {
    let x = u32::from_str_radix(input, 16).map_err(|e| {
        nom::Err::Error(VerboseError::from_external_error(
            input,
            ErrorKind::HexDigit,
            e,
        ))
    })?;
    Ok((input, x))
}

fn parse_literal_escape(input: &str) -> Res<char> {
    context(
        "literal escape",
        alt((
            char('\\'),
            char('"'),
            value('\n', char('n')),
            value('\r', char('r')),
            value('\t', char('t')),
            parse_escaped_unicode,
        )),
    )(input)
}

fn parse_char_class_escape(input: &str) -> Res<char> {
    context(
        "char class escape",
        alt((parse_literal_escape, char('-'), char(']'))),
    )(input)
}

fn parse_char_class<'a>(input: &'a str) -> Res<'a, CharClass> {
    context(
        "char class",
        wsc(delimited(
            char('['),
            cut(|input| {
                // caret or dash have special meaning as first character.
                let (input, first) = opt(alt((char('^'), char('-'))))(input)?;
                // todo: if the next character is a dash, this is a char range that starts with
                // dash :/

                // parse all the char ranges
                let (input, mut char_ranges) = many0(parse_char_range)(input)?;

                // if the first character is a caret, the char class is negated.
                // if the first character is a dash, it's meant literally.
                let mut negated = false;
                match first {
                    Some('^') => negated = true,
                    Some('-') => char_ranges.push(CharRange::Single('-')),
                    None => {}
                    _ => unreachable!(),
                }

                Ok((
                    input,
                    CharClass {
                        char_ranges,
                        negated,
                    },
                ))
            }),
            cut(char(']')),
        )),
    )(input)
}

fn parse_char_class_char(input: &str) -> Res<char> {
    let (mut input, c) = none_of("]-\r\n")(input)?;

    let c = match c {
        '\\' => {
            let c;
            (input, c) = parse_char_class_escape(input)?;
            c
        }
        _ => c,
    };

    Ok((input, c))
}

fn parse_char_range(input: &str) -> Res<CharRange> {
    let (input, first) = parse_char_class_char(input)?;
    let (mut input, dash) = opt(char('-'))(input)?;
    let char_range = if dash.is_some() {
        let second;
        (input, second) = parse_char_class_char(input)?;
        CharRange::Range {
            start: first,
            end: second,
        }
    }
    else {
        CharRange::Single(first)
    };

    Ok((input, char_range))
}

fn parse_parenthesis<'a>(input: &'a str) -> Res<'a, Alternatives> {
    delimited(wsc(char('(')), cut(parse_alternatives), cut(wsc(char(')'))))(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::ast::CharRange;

    #[test]
    fn it_parses_char_ranges() {
        assert_eq!(
            parse_char_range("a-z").unwrap().1,
            CharRange::Range {
                start: 'a',
                end: 'z'
            }
        );

        assert_eq!(
            parse_char_range("a-\\-").unwrap().1,
            CharRange::Range {
                start: 'a',
                end: '-'
            }
        );

        assert_eq!(parse_char_range("a").unwrap().1, CharRange::Single('a'));
    }

    #[test]
    fn it_fails_for_incomplete_char_range() {
        assert!(parse_char_range("a-").is_err(),);
    }

    #[test]
    fn it_parses_char_classes() {
        assert_eq!(
            parse_char_class("[a-zA-Z]").unwrap().1,
            CharClass {
                char_ranges: vec![
                    CharRange::Range {
                        start: 'a',
                        end: 'z'
                    },
                    CharRange::Range {
                        start: 'A',
                        end: 'Z'
                    },
                ],
                negated: false,
            }
        );

        assert_eq!(
            parse_char_class("[^a-zA-Z]").unwrap().1,
            CharClass {
                char_ranges: vec![
                    CharRange::Range {
                        start: 'a',
                        end: 'z'
                    },
                    CharRange::Range {
                        start: 'A',
                        end: 'Z'
                    },
                ],
                negated: true,
            }
        );

        assert_eq!(
            parse_char_class("[a-\\]]").unwrap().1,
            CharClass {
                char_ranges: vec![CharRange::Range {
                    start: 'a',
                    end: ']'
                },],
                negated: false,
            }
        );
    }

    #[test]
    fn it_parses_literals() {
        assert_eq!(
            parse_literal("\"hello world\"").unwrap().1,
            Literal("hello world".to_owned())
        );

        assert_eq!(
            parse_literal("\"\\\"\\r\\n\\t\\\\\"").unwrap().1,
            Literal("\"\r\n\t\\".to_owned())
        );

        assert_eq!(
            parse_literal("\"\\x611\\u00611\\U000000611\"").unwrap().1,
            Literal("a1a1a1".to_owned())
        );

        assert!(parse_literal("\"\\x1\"").is_err());
        assert!(parse_literal("\"\\u123\"").is_err());
        assert!(parse_literal("\"\\U1234567\"").is_err());

        assert!(parse_literal("\"\\a\"").is_err());
        assert!(parse_literal("\"hello\nworld\"").is_err());
    }

    #[test]
    fn it_parses_sequences() {
        assert_eq!(
            parse_sequence("hello").unwrap().1,
            Sequence(vec![Qualified {
                term: Term::Rule("hello".into()),
                qualifier: None
            },])
        );

        assert_eq!(
            parse_sequence(" hello ").unwrap().1,
            Sequence(vec![Qualified {
                term: Term::Rule("hello".into()),
                qualifier: None
            },])
        );

        assert_eq!(
            parse_sequence("\"hello world\" hello world").unwrap().1,
            Sequence(vec![
                Qualified {
                    term: Term::Literal(Literal("hello world".to_owned())),
                    qualifier: None
                },
                Qualified {
                    term: Term::Rule("hello".into()),
                    qualifier: None
                },
                Qualified {
                    term: Term::Rule("world".into()),
                    qualifier: None
                },
            ])
        );

        assert_eq!(
            parse_sequence("\"hello world\"? hello* world+").unwrap().1,
            Sequence(vec![
                Qualified {
                    term: Term::Literal(Literal("hello world".to_owned())),
                    qualifier: Some(Qualifier::Optional)
                },
                Qualified {
                    term: Term::Rule("hello".into()),
                    qualifier: Some(Qualifier::Many0)
                },
                Qualified {
                    term: Term::Rule("world".into()),
                    qualifier: Some(Qualifier::Many1)
                },
            ])
        );
    }

    #[test]
    fn it_parses_alternatives() {
        assert_eq!(
            parse_alternatives("\"hello world\"?").unwrap().1,
            Alternatives(vec![Sequence(vec![Qualified {
                term: Term::Literal(Literal("hello world".to_owned())),
                qualifier: Some(Qualifier::Optional)
            }]),])
        );

        assert_eq!(
            parse_alternatives("\"hello world\"? | hello* foo | world+ bar")
                .unwrap()
                .1,
            Alternatives(vec![
                Sequence(vec![Qualified {
                    term: Term::Literal(Literal("hello world".to_owned())),
                    qualifier: Some(Qualifier::Optional)
                }]),
                Sequence(vec![
                    Qualified {
                        term: Term::Rule("hello".into()),
                        qualifier: Some(Qualifier::Many0)
                    },
                    Qualified {
                        term: Term::Rule("foo".into()),
                        qualifier: None
                    },
                ]),
                Sequence(vec![
                    Qualified {
                        term: Term::Rule("world".into()),
                        qualifier: Some(Qualifier::Many1)
                    },
                    Qualified {
                        term: Term::Rule("bar".into()),
                        qualifier: None
                    },
                ]),
            ])
        );
    }

    #[test]
    fn it_parses_grammars() {
        assert_eq!(
            parse_grammar_complete(
                r#"
root ::= foo bar
foo ::= "foo"
bar ::= "bar"
            "#
            )
            .unwrap()
            .1,
            Grammar(vec![
                Production {
                    lhs: "root".into(),
                    rhs: Alternatives(vec![Sequence(vec![
                        Qualified {
                            term: Term::Rule("foo".into()),
                            qualifier: None
                        },
                        Qualified {
                            term: Term::Rule("bar".into()),
                            qualifier: None
                        },
                    ])])
                },
                Production {
                    lhs: "foo".into(),
                    rhs: Alternatives(vec![Sequence(vec![Qualified {
                        term: Term::Literal("foo".into()),
                        qualifier: None
                    },])])
                },
                Production {
                    lhs: "bar".into(),
                    rhs: Alternatives(vec![Sequence(vec![Qualified {
                        term: Term::Literal("bar".into()),
                        qualifier: None
                    },])])
                },
            ])
        );
    }

    #[test]
    fn it_parses_multiline_productions() {
        let s = r#"
root ::= (
    # it must start with the characters "1. " followed by a sequence
    # of characters that match the `move` rule, followed by a space, followed
    # by another move, and then a newline
    "1. " move " " move "\n"

    # it's followed by one or more subsequent moves, numbered with one or two digits
    ([1-9] [0-9]? ". " move " " move "\n")+
)
        "#;
        parse_production(s).unwrap();
    }

    #[test]
    fn it_parses_chess_example() {
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

        parse_grammar_complete(s).unwrap();
    }

    #[test]
    fn it_parses_json_example() {
        let s = r#"
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
    "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
    )? "}" ws

array  ::=
    "[" ws (
            value
    ("," ws value)*
    )? "]" ws

string ::=
    "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
    )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?        
        "#;

        parse_grammar_complete(s).unwrap();
    }
}
