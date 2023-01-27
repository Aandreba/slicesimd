//! Mostly taken from [`here`](https://stackoverflow.com/a/35270026)

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::{cell::UnsafeCell, mem::MaybeUninit};
#[allow(unused_imports)]
use docfg::docfg;

#[cfg(feature = "std")]
thread_local! {
    static COMPUTE_SPACE: UnsafeCell<Vec<u64>> = UnsafeCell::new(Vec::new());
}

macro_rules! impl_reduce_add {
    (
        $t:ident as $fn:ident + $spaced:ident {
            $(#[cfg($meta128:meta)])? $intr128:ident with $load128:ident,
            $(#[cfg($meta256:meta)])? $intr256:ident with $load256:ident,
            $(#[cfg($meta512:meta)])? $intr512:ident with $load512:ident
        }
    ) => {
        $(#[docfg($meta128)])?
        pub fn $fn (mut iter: &mut [$t]) -> $t {
            const SIMD128_LEN: usize = 16 / core::mem::size_of::<$t>();
            $(#[cfg($meta256)])?
            const SIMD256_LEN: usize = 32 / core::mem::size_of::<$t>();
            $(#[cfg($meta512)])?
            const SIMD512_LEN: usize = 64 / core::mem::size_of::<$t>();

            unsafe {
                $(#[cfg($meta512)])?
                loop {
                    let div = iter.len() / SIMD512_LEN;
                    if div == 0 {
                        break;
                    }
        
                    for i in 0..div {
                        let vec = $load512(iter.as_ptr().add(SIMD512_LEN * i));
                        *iter.get_unchecked_mut(i) = $intr512(vec)
                    }
        
                    *iter.get_unchecked_mut(div) = $fn(&mut iter[(SIMD512_LEN * div)..]);
                    iter = &mut iter[..=div];
                }
        
                $(#[cfg($meta256)])?
                loop {
                    let div = iter.len() / SIMD256_LEN;
                    if div == 0 {
                        break;
                    }
        
                    for i in 0..div {
                        let vec = $load256(iter.as_ptr().add(SIMD256_LEN * i));
                        *iter.get_unchecked_mut(i) = $intr512(vec)
                    }
        
                    *iter.get_unchecked_mut(div) = $fn(&mut iter[(SIMD256_LEN * div)..]);
                    iter = &mut iter[..=div];
                }
        
                loop {
                    let div = iter.len() / SIMD128_LEN;
                    if div == 0 {
                        return iter.iter().sum::<$t>();
                    }
        
                    for i in 0..div {
                        let vec = $load128(iter.as_ptr().add(SIMD128_LEN * i));
                        *iter.get_unchecked_mut(i) = $intr128(vec)
                    }
        
                    *iter.get_unchecked_mut(div) = iter[(SIMD128_LEN * div)..].into_iter().sum::<$t>();
                    iter = &mut iter[..=div];
                }
            }
        }

        $(#[docfg($meta128)])?
        pub fn $spaced (iter: &[$t], space: &mut [MaybeUninit<$t>]) -> $t {
            const SIMD128_LEN: usize = 16 / core::mem::size_of::<$t>();
            $(#[cfg($meta256)])?
            const SIMD256_LEN: usize = 32 / core::mem::size_of::<$t>();
            $(#[cfg($meta512)])?
            const SIMD512_LEN: usize = 64 / core::mem::size_of::<$t>();

            $(#[cfg($meta512)])?
            #[inline]
            unsafe fn spaced_512<'a> (iter: &[$t], space: &'a mut [$t]) -> Option<&'a mut [$t]> {
                let div = iter.len() / SIMD512_LEN;
                if div == 0 {
                    return None
                }
    
                for i in 0..div {
                    let vec = $load512(iter.as_ptr().add(SIMD512_LEN * i));
                    space[i] = $intr512(vec)
                }

                if let Some(chunk) = spaced_256(&iter[(SIMD512_LEN * div)..], space) {
                    space[div] = $fn(chunk);
                } else {
                    space[div] = spaced_128(&iter[(SIMD512_LEN * div)..], space);
                }

                return Some(&mut space[..=div])
            }

            $(#[cfg($meta256)])?
            #[inline]
            unsafe fn spaced_256<'a> (iter: &[$t], space: &'a mut [$t]) -> Option<&'a mut [$t]> {
                let div = iter.len() / SIMD256_LEN;
                if div == 0 {
                    return None
                }
    
                for i in 0..div {
                    let vec = $load256(iter.as_ptr().add(SIMD256_LEN * i));
                    space[i] = $intr256(vec)
                }

                space[div] = spaced_128(&iter[(SIMD256_LEN * div)..], space);
                return Some(&mut space[..=div])
            }

            unsafe fn spaced_128 (iter: &[$t], space: &mut [$t]) -> $t {
                let div = iter.len() / SIMD128_LEN;
                if div == 0 {
                    return iter.iter().sum()
                }
    
                for i in 0..div {
                    let vec = $load128(iter.as_ptr().add(SIMD128_LEN * i));
                    space[i] = $intr128(vec)
                }

                space[div] = iter[(SIMD128_LEN * div)..].iter().sum();
                return $fn(&mut space[..=div])
            }

            unsafe {
                // SAFETY: Garbage values aren't a problem, we'll never read them
                let space = core::slice::from_raw_parts_mut::<$t>(space.as_mut_ptr().cast(), space.len());

                $(#[cfg($meta512)])?
                if let Some(chunk) = spaced_512(iter, space) {
                    return $fn(chunk)
                }

                $(#[cfg($meta256)])?
                if let Some(chunk) = spaced_256(iter, space) {
                    return $fn(chunk)
                }
        
                return spaced_128(iter, space)
            }
        }
    };
}

impl_reduce_add! {
    f32 as reduce_add_f32_in_place + reduce_add_f32_with_space {
        f32x4_reduce_add with _mm_loadu_ps,
        #[cfg(target_feature = "avx")]
        f32x8_reduce_add with _mm256_loadu_ps,
        #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
        f32x16_reduce_add with _mm512_loadu_ps
    }
}

// #[docfg(feature = "std")]
// pub fn reduce_add(iter: &[f32]) -> f32 {
//     COMPUTE_SPACE.with(|x| unsafe {
//         const DELTA: usize = core::mem::size_of::<u64>() / core::mem::size_of::<f32>();
//         let space = &mut *x.get();
//         space.clear(); // avoid copying previous values if resizing
//         space.reserve((DELTA - 1) + iter.len() / DELTA);
// 
//         reduce_add_with_space(iter, core::slice::from_raw_parts_mut(space.as_mut_ptr(), (DELTA - 1) + space.capacity() / DELTA))
//     })
// }

/* FLOATS */
#[inline]
fn f32x4_reduce_add(v: __m128) -> f32 {
    unsafe {
        // [ C D | A B ]
        #[cfg(target_feature = "sse3")]
        let shuf = _mm_movehdup_ps(v);
        #[cfg(not(target_feature = "sse3"))]
        let shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        // sums = [ D+C C+D | B+A A+B ]
        let sums = _mm_add_ps(v, shuf);
        //  [   C   D | D+C C+D ]  // let the compiler avoid a mov by reusing shuf
        let shuf = _mm_movehl_ps(shuf, sums);
        let sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }
}

#[cfg(target_feature = "avx")]
#[inline]
fn f32x8_reduce_add(v: __m256) -> f32 {
    unsafe {
        let vlow = _mm256_castps256_ps128(v);
        // high 128
        let vhigh = _mm256_extractf128_ps(v, 1);
        // add the low 128
        let vlow = _mm_add_ps(vlow, vhigh);
        // and inline the sse3 version, which is optimal for AVX
        return f32x4_reduce_add(vlow);
    }
}

#[cfg(all(feature = "nightly", target_feature = "avx512f"))]
#[inline]
fn f32x16_reduce_add(v: __m512) -> f32 {
    unsafe {
        // [ C D | A B ]
        let shuf = _mm512_shuffle_f32x4(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        // sums = [ D+C C+D | B+A A+B ]
        let sums = _mm512_add_ps(v, shuf);
        //  [   C   D | D+C C+D ]  // let the compiler avoid a mov by reusing shuf
        let shuf = _mm512_shuffle_f32x4(shuf, sums, _MM_SHUFFLE(0, 1, 4, 5));
        let sums = _mm512_add_ps(sums, shuf);
        return f32x4_reduce_add(_mm512_extractf32x4_ps(sums, 0));
    }
}

/* DOUBLES */
#[cfg(target_feature = "sse2")]
#[inline]
fn f64x2_reduce_add(vd: __m128d) -> f64 {
    unsafe {
        // don't worry, we only use addSD, never touching the garbage bits with an FP add
        #[allow(invalid_value)]
        let undef = core::mem::MaybeUninit::uninit().assume_init();
        // there is no movhlpd
        let shuftmp = _mm_movehl_ps(undef, _mm_castpd_ps(vd));
        let shuf = _mm_castps_pd(shuftmp);
        return _mm_cvtsd_f64(_mm_add_sd(vd, shuf));
    }
}

#[cfg(target_feature = "avx")]
#[inline]
fn f64x4_reduce_add(vd: __m256d) -> f64 {
    unsafe {
        // [ C D | A B ]
        let shuf = _mm256_shuffle_pd(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        // sums = [ D+C C+D | B+A A+B ]
        let sums = _mm256_add_pd(v, shuf);
        //  [   C   D | D+C C+D ]  // let the compiler avoid a mov by reusing shuf
        let shuf = _mm256_shuffle_pd(shuf, sums, _MM_SHUFFLE(0, 1, 4, 5));
        let sums = _mm256_add_pd(sums, shuf);
        return _mm256_cvtsd_f64(sums);
    }
}

#[cfg(all(feature = "nightly", target_feature = "avx512f"))]
#[inline]
fn f64x8_reduce_add(v: __m512d) -> f64 {
    unsafe {
        let vlow = _mm512_castpd512_pd256(v);
        // high 128
        let vhigh = _mm512_extractf64x4_pd(v, 1);
        // add the low 128
        let vlow = _mm256_add_pd(vlow, vhigh);
        // and inline the sse3 version, which is optimal for AVX
        return f64x4_reduce_add(vlow);
    }
}

/* INT 32 */
#[cfg(target_feature = "sse2")]
#[inline]
fn i32x4_reduce_add(x: __m128i) -> i32 {
    unsafe {
        #[cfg(target_feature = "avx")]
        let hi64 = _mm_unpackhi_epi64(x, x);           // 3-operand non-destructive AVX lets us save a byte without needing a mov
        #[cfg(not(target_feature = "avx"))]
        let hi64 = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2));
        let sum64 = _mm_add_epi32(hi64, x);
        let hi32 = _mm_shufflelo_epi16(sum64, _MM_SHUFFLE(1, 0, 3, 2));    // Swap the low two elements
        let sum32 = _mm_add_epi32(sum64, hi32);
        return _mm_cvtsi128_si32(sum32);       // SSE2 movd
        //return _mm_extract_epi32(hl, 0);     // SSE4, even though it compiles to movd instead of a literal pextrd r32,xmm,0
    }
}

#[cfg(target_feature = "avx")]
#[inline]
fn i32x8_reduce_add(v: __m256i) -> i32 {
    unsafe {
        let vlow = _mm256_castsi256_si128(v);
        // high 128
        let vhigh = _mm256_extracti128_si256(v, 1);
        // add the low 128
        let vlow = _mm_add_epi32(vlow, vhigh);
        // and inline the sse3 version, which is optimal for AVX
        return i32x4_reduce_add(vlow);
    }
}

#[cfg(all(feature = "nightly", target_feature = "avx512f"))]
#[inline]
fn i32x16_reduce_add(v: __m512i) -> i32 {
    unsafe {
        // [ C D | A B ]
        let shuf = _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        // sums = [ D+C C+D | B+A A+B ]
        let sums = _mm512_add_epi32(v, shuf);
        //  [   C   D | D+C C+D ]  // let the compiler avoid a mov by reusing shuf
        let shuf = _mm512_shuffle_i32x4(shuf, sums, _MM_SHUFFLE(0, 1, 4, 5));
        let sums = _mm512_add_epi32(sums, shuf);
        return i32x4_reduce_add(_mm512_extracti32x4_epi32(sums, 0));
    }
}

#[inline]
#[allow(non_snake_case)]
pub const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[cfg(all(test, feature = "std"))]
mod tests {
    macro_rules! assert_feq {
        ($weight:expr, $lhs:expr, $rhs:expr) => {
            assert!(
                ($lhs - $rhs).abs() <= $weight * f32::EPSILON,
                "{} v. {}",
                $lhs,
                $rhs
            )
        };
    }

    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    use rand::{distributions::Standard, thread_rng, Rng};

    use crate::reduce_add_f32_in_place;

    #[test]
    fn test_f32() {
        // let mut v = thread_rng()
        //     .sample_iter(Standard)
        //     .take(3)
        //     .collect::<alloc::vec::Vec<_>>();
        // assert_eq!(reduce_add_in_place(&mut v), v.iter().sum());

        let mut v = thread_rng()
            .sample_iter(Standard)
            .take(4)
            .collect::<alloc::vec::Vec<_>>();

        assert_feq!(4.0, v.iter().sum::<f32>(), reduce_add_f32_in_place(&mut v));

        let mut v = thread_rng()
            .sample_iter(Standard)
            .take(5)
            .collect::<alloc::vec::Vec<_>>();
        assert_feq!(5.0, v.iter().sum::<f32>(), reduce_add_f32_in_place(&mut v));

        let mut v = thread_rng()
            .sample_iter(Standard)
            .take(10_000)
            .collect::<alloc::vec::Vec<_>>();
        assert_feq!(
            10_000f32,
            v.iter().sum::<f32>(),
            reduce_add_f32_in_place(&mut v)
        );

        let mut v = thread_rng()
            .sample_iter(Standard)
            .take(120_315)
            .collect::<alloc::vec::Vec<_>>();
        assert_feq!(
            120_315f32,
            v.iter().sum::<f32>(),
            reduce_add_f32_in_place(&mut v)
        );
    }

    #[test]
    fn test_f32() {
        use crate::reduce_add_in_place;

        // let mut v = thread_rng()
        //     .sample_iter(Standard)
        //     .take(3)
        //     .collect::<alloc::vec::Vec<_>>();
        // assert_eq!(reduce_add_in_place(&mut v), v.iter().sum());

        let mut v = thread_rng()
            .sample_iter(Standard)
            .take(4)
            .collect::<alloc::vec::Vec<_>>();

        assert_feq!(4.0, v.iter().sum::<f32>(), reduce_add_in_place(&mut v));

        let mut v = thread_rng()
            .sample_iter(Standard)
            .take(5)
            .collect::<alloc::vec::Vec<_>>();
        assert_feq!(5.0, v.iter().sum::<f32>(), reduce_add_in_place(&mut v));

        let mut v = thread_rng()
            .sample_iter(Standard)
            .take(10_000)
            .collect::<alloc::vec::Vec<_>>();
        assert_feq!(
            10_000f32,
            v.iter().sum::<f32>(),
            reduce_add_in_place(&mut v)
        );

        let mut v = thread_rng()
            .sample_iter(Standard)
            .take(120_315)
            .collect::<alloc::vec::Vec<_>>();
        assert_feq!(
            120_315f32,
            v.iter().sum::<f32>(),
            reduce_add_in_place(&mut v)
        );
    }
}
