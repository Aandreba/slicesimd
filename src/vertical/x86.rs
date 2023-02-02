#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::{ops::*};
use docfg::docfg;
use concat_idents::concat_idents;

macro_rules! impl_op {
    (
        $si:ident & $us:ident => $op:ident {
            $(#[cfg($meta128:meta)])? $ty128:ty: $intr128:ident & $load128:ident,
            $(#[cfg($meta256:meta)])? $ty256:ty: $intr256:ident & $load256:ident,
            $(#[cfg($meta512:meta)])? $ty512:ty: $intr512:ident & $load512:ident
        }
    ) => {
        impl_op! {
            $si => $op {
                $(#[cfg($meta128)])? $ty128: $intr128 & $load128,
                $(#[cfg($meta256)])? $ty256: $intr256 & $load256,
                $(#[cfg($meta512)])? $ty512: $intr512 & $load512
            }
        }

        impl_op! {
            $us => $op {
                $(#[cfg($meta128)])? $ty128: $intr128 & $load128,
                $(#[cfg($meta256)])? $ty256: $intr256 & $load256,
                $(#[cfg($meta512)])? $ty512: $intr512 & $load512
            }
        }
    };

    (
        $ty:ident => $op:ident {
            $(#[cfg($meta128:meta)])? $ty128:ty: $intr128:ident & $load128:ident,
            $(#[cfg($meta256:meta)])? $ty256:ty: $intr256:ident & $load256:ident,
            $(#[cfg($meta512:meta)])? $ty512:ty: $intr512:ident & $load512:ident
        }
    ) => {
        concat_idents!(f = $op, _assign_, $ty {
            $(#[docfg($meta128)])?
            #[inline]
            pub unsafe fn f(lhs: &mut [$ty], rhs: &[$ty]) {
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
                        return add_assign_512(lhs, rhs);
                    } else if #[cfg(all(always, $($meta256)?))] {
                        return add_assign_256(lhs, rhs);
                    } else {
                        return add_assign_128(lhs, rhs);
                    }
                }
            }
        });
    };
}

/* ADDITIONS */
impl_op! {
    f32 => add {
        __m128: _mm_add_ps & _mm_loadu_ps,
        #[cfg(target_feature = "avx")]
        __m256: _mm256_add_ps & _mm256_loadu_ps,
        #[cfg(target_feature = "avx512f")]
        __m512: _mm512_add_ps & _mm512_loadu_ps
    }
}

impl_op! {
    f64 => add {
        #[cfg(target_feature = "sse2")]
        __m128d: _mm_add_pd & _mm_loadu_pd,
        #[cfg(target_feature = "avx")]
        __m256d: _mm256_add_pd & _mm256_loadu_pd,
        #[cfg(target_feature = "avx512f")]
        __m512d: _mm512_add_pd & _mm512_loadu_pd
    }
}

impl_op! {
    i64 & u64 => add {
        #[cfg(target_feature = "sse2")]
        __m128i: _mm_add_epi64 & _mm_loadu_si64,
        #[cfg(target_feature = "avx2")]
        __m256i: _mm256_add_epi64 & _mm256_loadu_si256,
        #[cfg(target_feature = "avx512f")]
        __m512i: _mm512_add_epi64 & _mm512_loadu_si64
    }
}

impl_op! {
    i32 & u32 => add {
        #[cfg(target_feature = "sse2")]
        __m128i: _mm_add_epi32 & _mm_loadu_si64,
        #[cfg(target_feature = "avx2")]
        __m256i: _mm256_add_epi32 & _mm256_loadu_si256,
        #[cfg(target_feature = "avx512f")]
        __m512i: _mm512_add_epi32 & _mm512_loadu_si64
    }
}

impl_op! {
    i16 & u16 => add {
        #[cfg(target_feature = "sse2")]
        __m128i: _mm_add_epi16 & _mm_loadu_si64,
        #[cfg(target_feature = "avx2")]
        __m256i: _mm256_add_epi16 & _mm256_loadu_si256,
        #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
        __m512i: _mm512_add_epi16 & _mm512_loadu_si64
    }
}

impl_op! {
    i8 & u8 => add {
        #[cfg(target_feature = "sse2")]
        __m128i: _mm_add_epi8 & _mm_loadu_si64,
        #[cfg(target_feature = "avx2")]
        __m256i: _mm256_add_epi8 & _mm256_loadu_si256,
        #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
        __m512i: _mm512_add_epi8 & _mm512_loadu_si64
    }
}

#[cfg(test)]
mod tests {
    use crate::VerticalAdd;

    #[test]
    fn f32 () {
        let mut lhs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let rhs = [6.0, 7.0, 8.0, 9.0, 10.0];
        
        lhs.add_assign(&rhs);
        println!("{lhs:?}")
    }
}