#[repr(transparent)]
pub struct Chain<A, B> (core::iter::Chain<A, B>);

impl<T, A: Iterator<Item = T>, B: Iterator<Item = T>> Chain<A, B> {
    #[inline]
    pub fn new (left: A, right: B) -> Self {
        return Self(left.chain(right))
    }
}

impl<T, A: Iterator<Item = T>, B: Iterator<Item = T>> Iterator for Chain<A, B> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth(n)
    }
}

impl<T, A: ExactSizeIterator<Item = T>, B: ExactSizeIterator<Item = T>> ExactSizeIterator for Chain<A, B> {
    #[inline]
    fn len(&self) -> usize {
        let (lower, upper) = self.size_hint();
        debug_assert_eq!(upper, Some(lower));
        lower
    }
}