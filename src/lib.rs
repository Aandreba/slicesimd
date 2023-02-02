#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "nightly", feature(stdsimd))]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub(crate) mod sealed {
    use bytemuck::Pod;
    use num_traits::{NumOps, NumAssignOps};

    #[doc(hidden)]
    pub trait Slice {
        type Element: Pod + NumOps + NumAssignOps + for<'a> NumAssignOps<&'a Self::Element>;
        type Iter<'a>: Iterator<Item = &'a Self::Element> where Self: 'a;
        type IterMut<'a>: Iterator<Item = &'a mut Self::Element> where Self: 'a;
        
        fn len (&self) -> usize;
        fn iter (&self) -> Self::Iter<'_>;
        fn iter_mut (&mut self) -> Self::IterMut<'_>;
    }

    macro_rules! impl_slice {
        ($($t:ident),+) => {
            $(
                impl Slice for [$t] {
                    type Element = $t;
                    type Iter<'a> = core::slice::Iter<'a, $t>;
                    type IterMut<'a> = core::slice::IterMut<'a, $t>;
            
                    #[inline]
                    fn len (&self) -> usize { <[$t]>::len(self) }
                    #[inline]
                    fn iter (&self) -> Self::Iter<'_> { self.into_iter() }
                    #[inline]
                    fn iter_mut (&mut self) -> Self::IterMut<'_> { self.into_iter() }
                }  
            )+
        };
    }

    impl_slice! {
        u8, u16, u32, u64,
        i8, i16, i32, i64,
        f32, f64
    }
}

pub mod horizontal;
pub use horizontal::HorizontalSlice;

pub mod vertical;
pub use vertical::{VerticalAdd, VerticalSub, VerticalMul, VerticalDiv};

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

/// Functions to check currently enabled CPU extensions and functionality
pub mod checks {
    /// Returns `true` if the current target supports 64-bit SIMD types and operations, and `false` otherwise.
    #[inline]
    pub const fn is_simd_64() -> bool {
        return false;
    }

    /// Returns `true` if the current target supports 128-bit SIMD types and operations, and `false` otherwise.
    #[inline]
    pub const fn is_simd_128() -> bool {
        return is_x86_sse();
    }

    /// Returns `true` if the current target supports 256-bit SIMD types and operations, and `false` otherwise.
    #[inline]
    pub const fn is_simd_256() -> bool {
        return is_x86_avx();
    }

    /// Returns `true` if the current target supports 512-bit SIMD types and operations, and `false` otherwise.
    #[inline]
    pub const fn is_simd_512() -> bool {
        return is_x86_avx512();
    }

    /// Checks if the current platform is x86 (32-bit or 64-bit) and has support for SSE instructions.
    #[inline]
    pub const fn is_x86_sse() -> bool {
        return cfg!(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse"
        ));
    }

    /// Checks if the current platform is x86 (32-bit or 64-bit) and has support for SSE3 instructions.
    #[inline]
    pub const fn is_x86_sse3() -> bool {
        return cfg!(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse3"
        ));
    }

    /// Checks if the current platform is x86 (32-bit or 64-bit) and has support for SSE4 instructions.
    #[inline]
    pub const fn is_x86_sse4() -> bool {
        return cfg!(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse4"
        ));
    }

    /// Checks if the current platform is x86 (32-bit or 64-bit) and has support for AVX instructions.
    #[inline]
    pub const fn is_x86_avx() -> bool {
        return cfg!(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "avx"
        ));
    }

    /// Checks if the current platform is x86 (32-bit or 64-bit) and has support for AVX512 instructions.
    #[inline]
    pub const fn is_x86_avx512() -> bool {
        return cfg!(all(
            feature = "nightly",
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "avx512f"
        ));
    }

    /// Checks if the current platform is using naïve implementations of the algorithms.
    /// This is true when the `naive` feature is enabled, or as a fallback if no supported feature set is detected.
    #[inline]
    pub const fn is_naive() -> bool {
        return !is_x86_sse();
    }
}
