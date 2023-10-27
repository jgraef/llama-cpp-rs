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
