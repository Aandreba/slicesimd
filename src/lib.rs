#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "nightly", feature(stdsimd))]

#[cfg(feature = "alloc")]
pub(crate) extern crate alloc;

macro_rules! flat_mod {
    ($($i:ident),+) => {
        $(
            mod $i;
            pub use $i::*;
        )+
    }
}

cfg_if::cfg_if! {
    if #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse"))] {
        flat_mod! { x86 }
    } else {
        flat_mod! { naive }
    }
}

/* GENERIC */
#[inline]
pub const fn is_simd_64 () -> bool {
    return false
}

#[inline]
pub const fn is_simd_128 () -> bool {
    return is_x86_sse()
}

#[inline]
pub const fn is_simd_256 () -> bool {
    return is_x86_avx()
}

#[inline]
pub const fn is_simd_512 () -> bool {
    return is_x86_avx512()
}

/* X86 */
#[inline]
pub const fn is_x86_sse () -> bool {
    return cfg!(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse"))
}

#[inline]
pub const fn is_x86_sse3 () -> bool {
    return cfg!(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse3"))
}

#[inline]
pub const fn is_x86_avx () -> bool {
    return cfg!(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx"))
}

#[inline]
pub const fn is_x86_avx512 () -> bool {
    return cfg!(all(feature = "nightly", any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx512f"))
}

#[inline]
pub const fn is_naive_simd () -> bool {
    return !is_x86_sse()
}