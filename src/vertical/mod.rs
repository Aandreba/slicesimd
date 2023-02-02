use bytemuck::Pod;
use concat_idents::concat_idents;
use docfg::docfg;
use slicesimd_proc::simd_trait;

cfg_if::cfg_if! {
    if #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse"))] {
        mod x86;
        pub(crate) use x86::*;
    }
}

#[simd_trait]
pub trait VerticalAdd {
    fn add_assign (&mut self, rhs: &Self) {
        for (x, y) in self.iter_mut().zip(rhs) {
            x.add_assign(y)
        }
    }

    fn add_assign_checked (&mut self, rhs: &Self) -> bool;
    unsafe fn add_assign_unchecked(&mut self, rhs: &Self);
}

#[simd_trait]
pub trait VerticalSub {
    type Scalar: Pod;

    fn sub_assign (&mut self, rhs: &Self);
    fn sub_assign_checked (&mut self, rhs: &Self) -> bool;
    unsafe fn sub_assign_unchecked(&mut self, rhs: &Self);
}

#[simd_trait]
pub trait VerticalMul {
    type Scalar: Pod;

    fn mul_assign (&mut self, rhs: &Self);
    fn mul_assign_checked (&mut self, rhs: &Self) -> bool;
    unsafe fn mul_assign_unchecked(&mut self, rhs: &Self);
}

#[simd_trait]
pub trait VerticalDiv {
    type Scalar: Pod;

    fn div_assign (&mut self, rhs: &Self);
    fn div_assign_checked (&mut self, rhs: &Self) -> bool;
    unsafe fn div_assign_unchecked(&mut self, rhs: &Self);
}

macro_rules! impl_vert {
    (#[cfg($meta:meta)] $trait:ident as $fn:ident => $($t:ident),+) => {
        $(
            #[docfg($meta)]
            impl $trait for [$t] {
                type Scalar = $t;

                #[inline]
                fn $fn (&mut self, rhs: &Self) {
                    concat_idents!(f = $fn, _checked {
                        if !<Self as $trait>::f(self, rhs) {
                            panic!("Slice sizes don't match: {} v. {}", self.len(), rhs.len())
                        }
                    });

                }

                concat_idents!(f = $fn, _checked {
                    #[inline]
                    fn f (&mut self, rhs: &Self) -> bool {
                        if self.len() != rhs.len() { return false }
                        concat_idents!(f2 = $fn, _unchecked {
                            unsafe { <Self as $trait>::f2(self, rhs) }
                        });
                        return true
                    }
                });

                concat_idents!(f = $fn, _unchecked {
                    #[inline]
                    unsafe fn f (&mut self, rhs: &Self) {
                        debug_assert_eq!(self.len(), rhs.len());
                        concat_idents!(f2 = $fn, _, $t {
                            f2(self, rhs)
                        })
                    }
                });
            }
        )+
    };

    ($trait:ident as $fn:ident => $($t:ident),+) => {
        $(
            impl $trait for [$t] {
                type Scalar = $t;

                #[inline]
                fn $fn (&mut self, rhs: &Self) {
                    concat_idents!(f = $fn, _checked {
                        if !<Self as $trait>::f(self, rhs) {
                            panic!("Slice sizes don't match: {} v. {}", self.len(), rhs.len())
                        }
                    });

                }

                concat_idents!(f = $fn, _checked {
                    #[inline]
                    fn f (&mut self, rhs: &Self) -> bool {
                        if self.len() != rhs.len() { return false }
                        concat_idents!(f2 = $fn, _unchecked {
                            unsafe { <Self as $trait>::f2(self, rhs) }
                        });
                        return true
                    }
                });

                concat_idents!(f = $fn, _unchecked {
                    #[inline]
                    unsafe fn f (&mut self, rhs: &Self) {
                        debug_assert_eq!(self.len(), rhs.len());
                        concat_idents!(f2 = $fn, _, $t {
                            f2(self, rhs)
                        })
                    }
                });
            }
        )+
    };
}

/* ADDITION */
impl_vert!(SimdVerticalAdd as add_assign => f32);
impl_vert! {
    #[cfg(target_feature = "sse2") ]
    SimdVerticalAdd as add_assign =>
    u8, u16, u32, u64,
    i8, i16, i32, i64,
    f64
}

/* SUBTRACTION */
impl_vert!(SimdVerticalSub as sub_assign => f32);
impl_vert! {
    #[cfg(target_feature = "sse2") ]
    SimdVerticalSub as sub_assign =>
    u8, u16, u32, u64,
    i8, i16, i32, i64,
    f64
}

/* MULTIPLICATION */
impl_vert!(SimdVerticalMul as mul_assign => f32);
impl_vert! {
    #[cfg(target_feature = "sse2") ]
    SimdVerticalMul as mul_assign => f64
}

/* DIVISION */
impl_vert!(SimdVerticalDiv as div_assign => f32);
impl_vert! {
    #[cfg(target_feature = "sse2") ]
    SimdVerticalDiv as div_assign => f64
}