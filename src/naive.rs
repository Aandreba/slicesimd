use concat_idents::concat_idents;

macro_rules! impl_add {
    ($($t:ident),+) => {
        $(
            /* REDUCE_ADD */
            concat_idents!(f = reduce_add_, $t {
                #[inline]
                pub fn f (items: &[$t]) -> $t {
                    items.iter().sum()
                }
            });

            concat_idents!(f = reduce_add_, $t, _in_place {
                #[inline]
                pub fn f (items: &mut [$t]) -> $t {
                    items.iter().sum()
                }
            });

            concat_idents!(f = reduce_add_, $t, _in_space {
                #[inline]
                pub fn f (items: &[$t], _space: &mut [core::mem::MaybeUninit<$t>]) -> $t {
                    items.iter().sum()
                }
            });

            /* REDUCE_MUL */
            concat_idents!(f = reduce_mul_, $t {
                #[inline]
                pub fn f (items: &[$t]) -> $t {
                    items.iter().product()
                }
            });

            concat_idents!(f = reduce_mul_, $t, _in_place {
                #[inline]
                pub fn f (items: &mut [$t]) -> $t {
                    items.iter().product()
                }
            });

            concat_idents!(f = reduce_mul_, $t, _in_space {
                #[inline]
                pub fn f (items: &[$t], _space: &mut [core::mem::MaybeUninit<$t>]) -> $t {
                    items.iter().product()
                }
            });
        )+
    }
}

impl_add! {
    i32, f32, f64
}