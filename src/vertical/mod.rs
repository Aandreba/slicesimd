use bytemuck::Pod;
use concat_idents::concat_idents;
use docfg::docfg;

cfg_if::cfg_if! {
    if #[cfg(feature = "naive")] {
        mod naive;
        pub(crate) use naive::*;
    } else if #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse"))] {
        mod x86;
        pub(crate) use x86::*;
    } else {
        mod naive;
        pub(crate) use naive::*;
    }
}

pub trait VerticalAdd {
    type Scalar: Pod;

    fn add_assign (&mut self, rhs: &Self);
    fn add_assign_checked (&mut self, rhs: &Self) -> bool;
    unsafe fn add_assign_unchecked(&mut self, rhs: &Self);
}

pub trait VerticalSub {
    type Scalar: Pod;

    fn sub_assign (&mut self, rhs: &Self);
    fn sub_assign_checked (&mut self, rhs: &Self) -> bool;
    unsafe fn sub_assign_unchecked(&mut self, rhs: &Self);
}

macro_rules! impl_vert {
    (#[cfg($meta:meta)] $trait:ident as $fn:ident => $($t:ident),+) => {
        $(
            #[docfg($meta)]
            impl VerticalAdd for [$t] {
                type Scalar = $t;

                #[inline]
                fn add_assign (&mut self, rhs: &Self) {
                    if !self.add_assign_checked(rhs) {
                        panic!("Slice sizes don't match: {} v. {}", self.len(), rhs.len())
                    }
                }

                #[inline]
                fn add_assign_checked (&mut self, rhs: &Self) -> bool {
                    if self.len() != rhs.len() { return false }
                    unsafe { self.add_assign_unchecked(rhs) }
                    return true
                }

                #[inline]
                unsafe fn add_assign_unchecked (&mut self, rhs: &Self) {
                    debug_assert_eq!(self.len(), rhs.len());
                    concat_idents!(f = add_assign_, $t {
                        f(self, rhs)
                    })
                }
            }
        )+
    };

    ($trait:ident as $fn:ident => $($t:ident),+) => {
        $(
            impl $trait for [$t] {
                type Scalar = $t;

                #[inline]
                fn add_assign (&mut self, rhs: &Self) {
                    if !self.add_assign_checked(rhs) {
                        panic!("Slice sizes don't match: {} v. {}", self.len(), rhs.len())
                    }
                }

                #[inline]
                fn add_assign_checked (&mut self, rhs: &Self) -> bool {
                    if self.len() != rhs.len() { return false }
                    unsafe { self.add_assign_unchecked(rhs) }
                    return true
                }

                #[inline]
                unsafe fn add_assign_unchecked (&mut self, rhs: &Self) {
                    debug_assert_eq!(self.len(), rhs.len());
                    concat_idents!(f = add_assign_, $t {
                        f(self, rhs)
                    })
                }
            }
        )+
    };
}

impl_vert!(f32);

impl_vert! {
    #[cfg(target_feature = "sse2")]
    u8, u16, u32, u64,
    i8, i16, i32, i64,
    f64
}
