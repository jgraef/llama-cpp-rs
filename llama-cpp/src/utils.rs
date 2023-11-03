use std::iter::Peekable;

pub struct IsLast<I: Iterator>(Peekable<I>);

impl<I: Iterator> IsLast<I> {
    pub fn new(iter: I) -> Self {
        Self(iter.peekable())
    }
}

impl<I: Iterator> Iterator for IsLast<I> {
    type Item = (I::Item, bool);

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.0.next()?;
        let is_last = self.0.peek().is_none();
        Some((item, is_last))
    }
}

#[cfg(test)]
pub mod test {
    use std::cell::OnceCell;

    use crate::backend::{
        context::{
            Context,
            ContextParameters,
        },
        model::Model,
    };

    const MODEL: OnceCell<Model> = OnceCell::new();

    // we can share the model between tests
    pub fn model() -> Model {
        MODEL
            .get_or_init(|| {
                Model::load("../data/TinyLLama-v0.gguf", &Default::default(), |_| {})
                    .expect("failed to load model")
            })
            .clone()
    }

    pub fn context() -> Context {
        model().context(&ContextParameters {
            seed: Some(1234),
            ..Default::default()
        })
    }
}
