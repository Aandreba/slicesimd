#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::{ops::*, mem::MaybeUninit, ptr::addr_of};
use docfg::docfg;
use concat_idents::concat_idents;
use super::*;

macro_rules! impl_default {
    ($trait:ident => $($t:ty),+) => {
        $(
            impl $trait for [$t] {}
        )+
    };
}

macro_rules! impl_op {
    (
        $si:ident & $us:ident => $trait:ident as $op:ident {
            $(#[cfg($meta128:meta)])? $ty128:ty: $intr128:ident & $load128:ident,
            $(#[cfg($meta256:meta)])? $ty256:ty: $intr256:ident & $load256:ident,
            $(#[cfg($meta512:meta)])? $ty512:ty: $intr512:ident & $load512:ident
        }
    ) => {
        impl_op! {
            $si => $trait as $op {
                $(#[cfg($meta128)])? $ty128: $intr128 & $load128,
                $(#[cfg($meta256)])? $ty256: $intr256 & $load256,
                $(#[cfg($meta512)])? $ty512: $intr512 & $load512
            }
        }

        impl_op! {
            $us => $trait as $op {
                $(#[cfg($meta128)])? $ty128: $intr128 & $load128,
                $(#[cfg($meta256)])? $ty256: $intr256 & $load256,
                $(#[cfg($meta512)])? $ty512: $intr512 & $load512
            }
        }
    };

    (
        $ty:ident => $trait:ident as $op:ident {
            $(#[cfg($meta128:meta)])? $ty128:ty: $intr128:ident & $load128:ident,
            $(#[cfg($meta256:meta)])? $ty256:ty: $intr256:ident & $load256:ident,
            $(#[cfg($meta512:meta)])? $ty512:ty: $intr512:ident & $load512:ident
        }
    ) => {
        #[cfg(any(never, $(not($meta128))?))]
        $(#[cfg_attr(docsrs, doc(cfg(not($meta128))))])?
        impl $trait for [$ty] {}

        concat_idents!(r#trait = Simd, $trait {
            $(#[docfg($meta128)])?
            impl r#trait for [$ty] {
                concat_idents!(f = $op, _assign_unchecked {
                    #[inline]
                    unsafe fn f (&mut self, rhs: &Self) {
                        $(#[cfg($meta512)])?
                        #[inline]
                        unsafe fn add_assign_512(lhs: &mut [$ty], rhs: &[$ty]) {
                            const SIZE_DELTA: usize = core::mem::size_of::<$ty512>() / core::mem::size_of::<$ty>();
                            let (pre, simd, post) = lhs.align_to_mut::<$ty512>();
                    
                            // Add left size
                            add_assign_256(pre, &rhs[..pre.len()]);
                    
                            // Add SIMD aligned
                            let ptr = rhs.as_ptr().add(pre.len());
                            for i in 0..simd.len() {
                                let v = simd.get_unchecked_mut(i);
                                *v = $intr512(*v, $load512(ptr.add(SIZE_DELTA * i).cast()));
                            }
                    
                            // Add right size
                            let offset = pre.len() + simd.len() * SIZE_DELTA;
                            add_assign_256(post, &rhs[offset..]);
                        }
                        
                        $(#[cfg($meta256)])?
                        #[inline]
                        unsafe fn add_assign_256(lhs: &mut [$ty], rhs: &[$ty]) {
                            const SIZE_DELTA: usize = core::mem::size_of::<$ty256>() / core::mem::size_of::<$ty>();
                            let (pre, simd, post) = lhs.align_to_mut::<$ty256>();
                    
                            // Add left size
                            add_assign_128(pre, &rhs[..pre.len()]);
                    
                            // Add SIMD aligned
                            let ptr = rhs.as_ptr().add(pre.len());
                            for i in 0..simd.len() {
                                let v = simd.get_unchecked_mut(i);
                                *v = $intr256(*v, $load256(ptr.add(SIZE_DELTA * i).cast()));
                            }
                    
                            // Add right size
                            let offset = pre.len() + simd.len() * SIZE_DELTA;
                            add_assign_128(post, &rhs[offset..]);
                        }
                    
                        #[inline]
                        unsafe fn add_assign_128(lhs: &mut [$ty], rhs: &[$ty]) {
                            const SIZE_DELTA: usize = core::mem::size_of::<$ty128>() / core::mem::size_of::<$ty>();
                            let (pre, simd, post) = lhs.align_to_mut::<$ty128>();
                    
                            // Add left size
                            for i in 0..pre.len() {
                                pre.get_unchecked_mut(i).add_assign(rhs.get_unchecked(i));
                            }
                    
                            // Add SIMD aligned
                            let ptr = rhs.as_ptr().add(pre.len());
                            for i in 0..simd.len() {
                                let v = simd.get_unchecked_mut(i);
                                *v = $intr128(*v, $load128(ptr.add(SIZE_DELTA * i).cast()));
                            }
                    
                            // Add right size
                            let offset = pre.len() + simd.len() * SIZE_DELTA;
                            for i in 0..post.len() {
                                post.get_unchecked_mut(i)
                                    .add_assign(rhs.get_unchecked(offset + i));
                            }
                        }
                    
                        cfg_if::cfg_if! {
                            if #[cfg(all(always, $($meta512)?))] {
                                return add_assign_512(self, rhs);
                            } else if #[cfg(all(always, $($meta256)?))] {
                                return add_assign_256(self, rhs);
                            } else {
                                return add_assign_128(self, rhs);
                            }
                        }
                    }
                });
            }
        });
    };
}

/* ADDITIONS */
impl_op! {
    f32 => VerticalAdd as add {
        __m128: _mm_add_ps & _mm_loadu_ps,
        #[cfg(target_feature = "avx")]
        __m256: _mm256_add_ps & _mm256_loadu_ps,
        #[cfg(target_feature = "avx512f")]
        __m512: _mm512_add_ps & _mm512_loadu_ps
    }
}

impl_op! {
    f64 => VerticalAdd as add {
        #[cfg(target_feature = "sse2")]
        __m128d: _mm_add_pd & _mm_loadu_pd,
        #[cfg(target_feature = "avx")]
        __m256d: _mm256_add_pd & _mm256_loadu_pd,
        #[cfg(target_feature = "avx512f")]
        __m512d: _mm512_add_pd & _mm512_loadu_pd
    }
}

impl_op! {
    i64 & u64 => VerticalAdd as add {
        #[cfg(target_feature = "sse2")]
        __m128i: _mm_add_epi64 & _mm_loadu_si64,
        #[cfg(target_feature = "avx2")]
        __m256i: _mm256_add_epi64 & _mm256_loadu_si256,
        #[cfg(target_feature = "avx512f")]
        __m512i: _mm512_add_epi64 & _mm512_loadu_si64
    }
}

impl_op! {
    i32 & u32 => VerticalAdd as add {
        #[cfg(target_feature = "sse2")]
        __m128i: _mm_add_epi32 & _mm_loadu_si64,
        #[cfg(target_feature = "avx2")]
        __m256i: _mm256_add_epi32 & _mm256_loadu_si256,
        #[cfg(target_feature = "avx512f")]
        __m512i: _mm512_add_epi32 & _mm512_loadu_si64
    }
}

impl_op! {
    i16 & u16 => VerticalAdd as add {
        #[cfg(target_feature = "sse2")]
        __m128i: _mm_add_epi16 & _mm_loadu_si64,
        #[cfg(target_feature = "avx2")]
        __m256i: _mm256_add_epi16 & _mm256_loadu_si256,
        #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
        __m512i: _mm512_add_epi16 & _mm512_loadu_si64
    }
}

impl_op! {
    i8 & u8 => VerticalAdd as add {
        #[cfg(target_feature = "sse2")]
        __m128i: _mm_add_epi8 & _mm_loadu_si64,
        #[cfg(target_feature = "avx2")]
        __m256i: _mm256_add_epi8 & _mm256_loadu_si256,
        #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
        __m512i: _mm512_add_epi8 & _mm512_loadu_si64
    }
}

/* SUBTRACTIONS */
impl_op! {
    f32 => VerticalSub as sub {
        __m128: _mm_sub_ps & _mm_loadu_ps,
        #[cfg(target_feature = "avx")]
        __m256: _mm256_sub_ps & _mm256_loadu_ps,
        #[cfg(target_feature = "avx512f")]
        __m512: _mm512_sub_ps & _mm512_loadu_ps
    }
}

impl_op! {
    f64 => VerticalSub as sub {
        #[cfg(target_feature = "sse2")]
        __m128d: _mm_sub_pd & _mm_loadu_pd,
        #[cfg(target_feature = "avx")]
        __m256d: _mm256_sub_pd & _mm256_loadu_pd,
        #[cfg(target_feature = "avx512f")]
        __m512d: _mm512_sub_pd & _mm512_loadu_pd
    }
}

impl_op! {
    i64 & u64 => VerticalSub as sub {
        #[cfg(target_feature = "sse2")]
        __m128i: _mm_sub_epi64 & _mm_loadu_si64,
        #[cfg(target_feature = "avx2")]
        __m256i: _mm256_sub_epi64 & _mm256_loadu_si256,
        #[cfg(target_feature = "avx512f")]
        __m512i: _mm512_sub_epi64 & _mm512_loadu_si64
    }
}

impl_op! {
    i32 & u32 => VerticalSub as sub {
        #[cfg(target_feature = "sse2")]
        __m128i: _mm_sub_epi32 & _mm_loadu_si64,
        #[cfg(target_feature = "avx2")]
        __m256i: _mm256_sub_epi32 & _mm256_loadu_si256,
        #[cfg(target_feature = "avx512f")]
        __m512i: _mm512_sub_epi32 & _mm512_loadu_si64
    }
}

impl_op! {
    i16 & u16 => VerticalSub as sub {
        #[cfg(target_feature = "sse2")]
        __m128i: _mm_sub_epi16 & _mm_loadu_si64,
        #[cfg(target_feature = "avx2")]
        __m256i: _mm256_sub_epi16 & _mm256_loadu_si256,
        #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
        __m512i: _mm512_sub_epi16 & _mm512_loadu_si64
    }
}

impl_op! {
    i8 & u8 => VerticalSub as sub {
        #[cfg(target_feature = "sse2")]
        __m128i: _mm_sub_epi8 & _mm_loadu_si64,
        #[cfg(target_feature = "avx2")]
        __m256i: _mm256_sub_epi8 & _mm256_loadu_si256,
        #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
        __m512i: _mm512_sub_epi8 & _mm512_loadu_si64
    }
}

/* MULTIPLICATIONS */
impl_op! {
    f32 => VerticalMul as mul {
        __m128: _mm_mul_ps & _mm_loadu_ps,
        #[cfg(target_feature = "avx")]
        __m256: _mm256_mul_ps & _mm256_loadu_ps,
        #[cfg(target_feature = "avx512f")]
        __m512: _mm512_mul_ps & _mm512_loadu_ps
    }
}

impl_op! {
    f64 => VerticalMul as mul {
        #[cfg(target_feature = "sse2")]
        __m128d: _mm_mul_pd & _mm_loadu_pd,
        #[cfg(target_feature = "avx")]
        __m256d: _mm256_mul_pd & _mm256_loadu_pd,
        #[cfg(target_feature = "avx512f")]
        __m512d: _mm512_mul_pd & _mm512_loadu_pd
    }
}

impl_default! {
    VerticalMul =>
    u8, u16, u32, u64,
    i8, i16, i32, i64
}

/* DIVISIONS */
impl_op! {
    f32 => VerticalDiv as div {
        __m128: _mm_div_ps & _mm_loadu_ps,
        #[cfg(target_feature = "avx")]
        __m256: _mm256_div_ps & _mm256_loadu_ps,
        #[cfg(target_feature = "avx512f")]
        __m512: _mm512_div_ps & _mm512_loadu_ps
    }
}

impl_op! {
    f64 => VerticalDiv as div {
        #[cfg(target_feature = "sse2")]
        __m128d: _mm_div_pd & _mm_loadu_pd,
        #[cfg(target_feature = "avx")]
        __m256d: _mm256_div_pd & _mm256_loadu_pd,
        #[cfg(target_feature = "avx512f")]
        __m512d: _mm512_div_pd & _mm512_loadu_pd
    }
}

impl_default! {
    VerticalDiv =>
    u8, u16, u32, u64,
    i8, i16, i32, i64
}

#[inline]
unsafe fn add_in (lhs: &[f32], rhs: &[f32], result: &mut [MaybeUninit<f32>]) {
    const DELTA_SIZE: usize = core::mem::size_of::<__m128>() / core::mem::size_of::<f32>();
    
    let div = lhs.len() / DELTA_SIZE;
    let rem = lhs.len() % DELTA_SIZE;

    let lhs = lhs.as_ptr();
    let rhs = rhs.as_ptr();
    let result = result.as_mut_ptr().cast::<f32>();

    for i in 0..div {
        let offset = DELTA_SIZE * i;
        let v = _mm_add_ps(
            _mm_loadu_ps(lhs.add(offset)),
            _mm_loadu_ps(rhs.add(offset))
        );
        core::ptr::copy_nonoverlapping(
            addr_of!(v).cast(),
            result.add(offset),
            DELTA_SIZE
        )
    }

    let offset = DELTA_SIZE * div;
    for i in offset..(offset + rem) {
        result.add(i).write(*lhs.add(i) + *rhs.add(i))
    }
}