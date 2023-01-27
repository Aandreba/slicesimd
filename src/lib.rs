#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "nightly", feature(stdsimd))]
#![cfg_attr(docsrs, feature(doc_cfg))]

use bytemuck::Pod;
#[allow(unused_imports)]
use concat_idents::concat_idents;
use core::mem::MaybeUninit;
use docfg::docfg;

#[cfg(feature = "alloc")]
pub(crate) extern crate alloc;

#[allow(unused_macros)]
macro_rules! flat_mod {
    ($($i:ident),+) => {
        $(
            mod $i;
            pub use $i::*;
        )+
    }
}

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

pub trait SliceExt {
    type Scalar: Pod;

    /// Adds up all the values in the slice horizontally.
    ///
    /// Since this method doesn't have mutable access to it's target, it may use a thread local "compute space"
    /// to store the temporary results of the operations.
    ///
    /// If this method is called on a slice of integers (signed or unsigned), the operation will be done with wrapping addition.
    ///
    /// > # Note
    /// > When using naive mode, no compute space will be used
    /// 
    /// # Example
    /// ```rust
    /// use simdslice::*;
    ///
    /// let values = [1, 2, 3, 4, 5];
    /// assert_eq!(values.reduce_add(), 15);
    /// ```
    #[docfg(feature = "std")]
    fn reduce_add(&self) -> Self::Scalar;

    /// Adds up all the values in the slice horizontaly, using `space` to store temporary data.
    ///
    /// Since this method doesn't have mutable access to it's target, it may use `space` as a "compute space"
    /// to store the temporary results of the operations.
    ///
    /// If this method is called on a slice of integers (signed or unsigned), the operation will be done with wrapping addition.
    ///
    /// > # Note
    /// > When using naive mode, `space` will be used
    /// 
    /// # Example
    /// ```rust
    /// #![feature(maybe_uninit_uninit_array)]
    ///
    /// use slicesimd::*;
    /// use core::mem::MaybeUninit;
    ///
    /// let values = [1, 2, 3, 4, 5];
    /// let mut blank_space = MaybeUninit::uninit_array();
    /// let sum = values.reduce_add_in_place(&mut blank_space);
    ///
    /// assert_eq!(values[0], sum);
    /// assert_eq!(sum, 15);
    /// ```
    fn reduce_add_in_space(&self, space: &mut [MaybeUninit<Self::Scalar>]) -> Self::Scalar;

    /// Adds up all the values in the slice horizontally, storing temporary data in the same slice.
    ///
    /// Since this method has mutable access to it's target, it will use the target itself
    /// to store the temporary results of the operations.
    ///
    /// The resulting slice will have the result of the operation in it's first index, but the remaining values are undefined.
    ///
    /// If this method is called on a slice of integers (signed or unsigned), the operation will be done with wrapping addition.
    ///
    /// # Example
    /// ```rust
    /// use slicesimd::*;
    ///
    /// let mut values = [1, 2, 3, 4, 5];
    /// let sum = values.reduce_add_in_place();
    ///
    /// assert_eq!(values[0], sum);
    /// assert_eq!(sum, 15);
    /// ```
    fn reduce_add_in_place(&mut self) -> Self::Scalar;
}

macro_rules! impl_slice_ext {
    ($($t:ident),+) => {
        $(
            impl SliceExt for [$t] {
                type Scalar = $t;

                #[cfg(feature = "std")]
                #[inline]
                fn reduce_add (&self) -> Self::Scalar {
                    concat_idents!(f = reduce_add_, $t {
                        f(self)
                    })
                }

                #[inline]
                fn reduce_add_in_space (&self, space: &mut [MaybeUninit<$t>]) -> Self::Scalar {
                    concat_idents!(f = reduce_add_, $t, _in_space {
                        f(self, space)
                    })
                }

                #[inline]
                fn reduce_add_in_place (&mut self) -> Self::Scalar {
                    concat_idents!(f = reduce_add_, $t, _in_place {
                        f(self)
                    })
                }
            }
        )+
    };
}

impl_slice_ext! {
    i32,
    f32, f64
}

/* GENERIC */
#[inline]
pub const fn is_simd_64() -> bool {
    return false;
}

#[inline]
pub const fn is_simd_128() -> bool {
    return is_x86_sse();
}

#[inline]
pub const fn is_simd_256() -> bool {
    return is_x86_avx();
}

#[inline]
pub const fn is_simd_512() -> bool {
    return is_x86_avx512();
}

/* X86 */
#[inline]
pub const fn is_x86_sse() -> bool {
    return cfg!(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse"
    ));
}

#[inline]
pub const fn is_x86_sse3() -> bool {
    return cfg!(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse3"
    ));
}

#[inline]
pub const fn is_x86_sse4() -> bool {
    return cfg!(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse4"
    ));
}

#[inline]
pub const fn is_x86_avx() -> bool {
    return cfg!(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx"
    ));
}

#[inline]
pub const fn is_x86_avx512() -> bool {
    return cfg!(all(
        feature = "nightly",
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx512f"
    ));
}

/* NAIVE */
#[inline]
pub const fn is_naive() -> bool {
    return cfg!(feature = "naive") || !is_x86_sse();
}
