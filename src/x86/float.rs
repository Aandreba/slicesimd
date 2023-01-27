//! Mostly taken from [`here`](https://stackoverflow.com/a/35270026)

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::cell::UnsafeCell;
use docfg::docfg;

#[docfg(feature = "std")]
pub fn reduce_add(iter: &[f32]) -> f32 {
    thread_local! {
        static SPACE: UnsafeCell<Vec<u64>> = UnsafeCell::new(Vec::new());
    }

    SPACE.with(|x| reduce_add_with_space(iter, unsafe { &mut *x.get() }))
}

#[docfg(feature = "alloc")]
pub fn reduce_add_with_space(iter: &[f32], space: &mut alloc::vec::Vec<u64>) -> f32 {
    const DELTA: usize = core::mem::size_of::<u64>() / core::mem::size_of::<f32>(); // 2

    space.clear(); // avoid copying previous values
    space.reserve((DELTA - 1) + iter.len() / DELTA);

    unsafe {
        let mut space =
            core::slice::from_raw_parts_mut::<f32>(space.as_mut_ptr().cast(), iter.len());

        #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
        loop {
            let div = iter.len() / 16;
            if div == 0 {
                break;
            }

            for i in 0..div {
                let vec = _mm512_loadu_ps(iter.as_ptr().add(16 * i));
                *space.get_unchecked_mut(i) = f32x16_reduce_add(vec)
            }

            *space.get_unchecked_mut(div) = iter[(16 * div)..].into_iter().sum::<f32>();
            space = &mut space[..=div];
        }

        #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
        let iter = space;

        #[cfg(target_feature = "avx")]
        loop {
            let div = iter.len() / 8;
            if div == 0 {
                break;
            }

            for i in 0..div {
                let vec = _mm256_loadu_ps(iter.as_ptr().add(8 * i));
                *space.get_unchecked_mut(i) = f32x8_reduce_add(vec)
            }

            *space.get_unchecked_mut(div) = iter[(8 * div)..].into_iter().sum::<f32>();
            space = &mut space[..=div];
        }

        #[cfg(target_feature = "avx")]
        let iter = space;

        loop {
            let div = iter.len() / 4;
            if div == 0 {
                return iter.iter().sum::<f32>();
            }

            for i in 0..div {
                let vec = _mm_loadu_ps(iter.as_ptr().add(4 * i));
                *space.get_unchecked_mut(i) = f32x4_reduce_add(vec);
            }

            *space.get_unchecked_mut(div) = iter[(4 * div)..].into_iter().sum::<f32>();
            space = &mut space[..=div];
        }
    }
}

pub fn reduce_add_in_place(mut iter: &mut [f32]) -> f32 {
    unsafe {
        #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
        loop {
            let div = iter.len() / 16;
            if div == 0 {
                break;
            }

            for i in 0..div {
                let vec = _mm512_loadu_ps(iter.as_ptr().add(16 * i));
                *iter.get_unchecked_mut(i) = f32x16_reduce_add(vec)
            }

            *iter.get_unchecked_mut(div) = iter[(16 * div)..].into_iter().sum::<f32>();
            iter = &mut iter[..=div];
        }

        #[cfg(target_feature = "avx")]
        loop {
            let div = iter.len() / 8;
            if div == 0 {
                break;
            }

            for i in 0..div {
                let vec = _mm256_loadu_ps(iter.as_ptr().add(8 * i));
                *iter.get_unchecked_mut(i) = f32x8_reduce_add(vec)
            }

            *iter.get_unchecked_mut(div) = iter[(8 * div)..].into_iter().sum::<f32>();
            iter = &mut iter[..=div];
        }

        loop {
            let div = iter.len() / 4;
            if div == 0 {
                return iter.iter().sum::<f32>();
            }

            for i in 0..div {
                let vec = _mm_loadu_ps(iter.as_ptr().add(4 * i));
                *iter.get_unchecked_mut(i) = f32x4_reduce_add(vec)
            }

            *iter.get_unchecked_mut(div) = iter[(4 * div)..].into_iter().sum::<f32>();
            iter = &mut iter[..=div];
        }
    }
}

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

/* UINT32 */
#[cfg(target_feature = "sse2")]
#[inline]
fn u32x4_reduce_add(v: __m128i) -> u32 {
    unsafe {
        // [ C D | A B ]
        let shuf = _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
        // sums = [ D+C C+D | B+A A+B ]
        let sums = _mm_add_epi32(v, shuf);
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

#[inline]
#[allow(non_snake_case)]
pub const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[cfg(test)]
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
    use rand::{distributions::Standard, random, thread_rng, Rng};

    #[cfg(feature = "alloc")]
    #[test]
    fn test_slice() {
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

    #[cfg(all(
        target_feature = "sse",
        not(target_feature = "sse3"),
        not(target_feature = "avx"),
        not(target_feature = "avx512f")
    ))]
    #[test]
    fn test_sse() {
        use super::f32x4_reduce_add;

        unsafe {
            let values = random::<[f32; 4]>();
            let simd = _mm_loadu_ps(values.as_ptr());
            assert!(f32::abs(f32x4_reduce_add(simd) - values.iter().sum::<f32>()) <= f32::EPSILON);
        }
    }

    #[cfg(all(
        target_feature = "sse",
        target_feature = "sse3",
        not(target_feature = "avx"),
        not(target_feature = "avx512f")
    ))]
    #[test]
    fn test_sse3() {
        use super::f32x4_reduce_add;

        unsafe {
            let values = random::<[f32; 4]>();
            let simd = _mm_loadu_ps(values.as_ptr());
            assert!(f32::abs(f32x4_reduce_add(simd) - values.iter().sum::<f32>()) <= f32::EPSILON);
        }
    }

    #[cfg(all(
        target_feature = "sse",
        target_feature = "sse3",
        target_feature = "avx",
        not(target_feature = "avx512f")
    ))]
    #[test]
    fn test_avx() {
        use super::f32x8_reduce_add;

        unsafe {
            let values = random::<[f32; 8]>();
            let simd = _mm256_loadu_ps(values.as_ptr());

            let lhs = f32x8_reduce_add(simd);
            let rhs = values.iter().sum::<f32>();
            assert!(f32::abs(lhs - rhs) <= 2.0 * f32::EPSILON, "{lhs} v. {rhs}");
        }
    }

    #[cfg(all(
        feature = "nightly",
        target_feature = "sse",
        target_feature = "sse3",
        target_feature = "avx",
        target_feature = "avx512f"
    ))]
    #[test]
    fn test_avx512() {
        use super::f32x16_reduce_add;

        unsafe {
            let values = random::<[f32; 16]>();
            let simd = _mm512_loadu_ps(values.as_ptr());

            let lhs = f32x16_reduce_add(simd);
            let rhs = values.iter().sum::<f32>();
            assert!(f32::abs(lhs - rhs) <= 4.0 * f32::EPSILON, "{lhs} v. {rhs}");
        }
    }
}
