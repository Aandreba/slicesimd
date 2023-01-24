#![no_std]
#![feature(stdsimd, trait_alias, iter_array_chunks)]

cfg_if::cfg_if! {
    if #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse"))] {
        mod x86;
        use x86::*;
    } else {
        mod naive;
        use naive::*;
    }
}

pub trait SliceExt {
    type Scalar;

    fn reduce_sum (self) -> Self::Scalar;
}