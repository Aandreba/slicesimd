use concat_idents::concat_idents;
use core::ops::*;

macro_rules! impl_vertical {
    ($($t:ident),+) => {
        $(
            concat_idents!(f = add_assign_, $t {
                #[inline]
                pub fn f (lhs: &mut [$t], rhs: &[$t]) {
                    for (x, y) in lhs.zip(rhs) {
                        x.add_assign(y)
                    }
                }
            });
        )+
    }
}

impl_vertical! {
    u8, u16, u32, u64,
    i8, i16, i32, i64,
    f32, f64
}
