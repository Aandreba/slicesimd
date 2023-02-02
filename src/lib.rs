#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "nightly", feature(stdsimd))]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub(crate) mod sealed {
    use bytemuck::Pod;

    pub trait Slice
    where
        for<'a> &'a Self: IntoIterator<Item = &'a Self::Element>,
        for<'a> &'a mut Self: IntoIterator<Item = &'a mut Self::Element>,
    {
        type Element: Pod;
    }

    impl<T> Slice for [T] {
        type Element = T;
    }
}

pub mod horizontal;
pub use horizontal::HorizontalSlice;

pub mod vertical;
pub use vertical::SimdVerticalAdd;

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
