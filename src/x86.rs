//! Mostly taken from [`here`](https://stackoverflow.com/a/35270026)

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::{iter::ArrayChunks, hint::unreachable_unchecked, ptr::{addr_of}};

pub fn reduce_add_iter (iter: &mut dyn ExactSizeIterator<Item = f32>) -> f32 {
    enum Status<'a> {
        #[cfg(target_feature = "avx")]
        Simd256 (ArrayChunks<&'a mut dyn ExactSizeIterator<Item = f32>, 8>),
        #[cfg(target_feature = "avx")]
        Simd128 (core::array::IntoIter<f32, 8>),
        #[cfg(not(target_feature = "avx"))]
        Simd128 (ArrayChunks<&'a mut dyn ExactSizeIterator<Item = f32>, 4>),
        Rem (core::array::IntoIter<f32, 4>)
    }

    struct SplitIter<'a> {
        iter: Option<Status<'a>>,
        rem: usize
    }

    impl<'a> Iterator for SplitIter<'a> {
        type Item = f32;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let iter_ptr = addr_of!(self.iter);

            if let Some(ref mut iter) = self.iter {
                #[cfg(target_feature = "avx")]
                if let Status::Simd256(ref mut iter256) = iter {
                    if let Some(x) = iter256.next() {
                        return unsafe { Some(f32x8_reduce_add(_mm256_load_ps(x.as_ptr()))) } 
                    } else {
                        unsafe {
                            drop(iter256);
                            let iter256 = match core::ptr::read(iter256) {
                                Some(Status::Simd256(x)) => x,
                                _ => unreachable_unchecked()
                            };

                            #[cfg(debug_assertions)]
                            let iter = iter256.into_remainder().unwrap();
                            #[cfg(not(debug_assertions))]
                            let iter = iter256.into_remainder().unwrap_unchecked();
                            core::ptr::write(iter_ptr.cast_mut(), Some(Status::Rem(iter)))
                        }
                    }
                }

                if let Status::Simd128(ref mut iter128) = iter {
                    if let Some(x) = iter128.next() {
                        return unsafe { Some(f32x4_reduce_add(_mm_load_ps(x.as_ptr()))) } 
                    } else {
                        unsafe {
                            drop(iter128);
                            let iter128 = match core::ptr::read(iter_ptr) {
                                Some(Status::Simd128(x)) => x,
                                _ => unreachable_unchecked()
                            };

                            #[cfg(debug_assertions)]
                            let iter = iter128.into_remainder().unwrap();
                            #[cfg(not(debug_assertions))]
                            let iter = iter128.into_remainder().unwrap_unchecked();
                            core::ptr::write(iter_ptr.cast_mut(), Some(Status::Rem(iter)))
                        }
                    }
                }
    
                if let Status::Rem(ref mut iter) = iter {
                    if let Some(v) = iter.next() {
                        return Some(v)
                    } else {
                        self.iter = None;
                    }
                }
            }

            return None
        }
    }

    impl ExactSizeIterator for SplitIter<'_> {
        #[inline]
        fn len(&self) -> usize {
            match self.iter {
                Some(Status::Simd128(ref x)) => x.len() + self.rem,
                Some(Status::Rem(ref x)) => x.len(),
                None => 0
            }
        }
    }

    let len = iter.len();
    match len {
        0 => return 0.0,
        1 => return unsafe { iter.next().unwrap_unchecked() },
        2 | 3 => return iter.sum(),
        _ => {} 
    }

    let mut iter = SplitIter {
        iter: Some(Status::Simd128(iter.array_chunks::<4>())),
        rem: (len * core::mem::size_of::<f32>()) % core::mem::size_of::<__m128>(),
    };

    return reduce_add(&mut iter);
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
        let shuf = _mm512_shuffle_f32x4(shuf, sums._MM_SHUFFLE(0, 1, 4, 5));
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
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    use crate::x86::reduce_add;

    #[test]
    fn test_slice () {
        let mut v = [1.0, 2.0, 3.0, 4.0].into_iter();
        assert_eq!(reduce_add(&mut v), 10.0);
    }

    #[cfg(all(
        target_feature = "sse",
        not(target_feature = "sse3"),
        not(target_feature = "avx"),
        not(target_feature = "avx512f")
    ))]
    #[test]
    fn test_sse() {
        use core::f32::consts::{E, LN_2, PI, SQRT_2};

        use super::f32x4_reduce_add;

        unsafe {
            let values = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
            assert_eq!(f32x4_reduce_add(values), 10.0);

            let values = _mm_set_ps(PI, E, SQRT_2, LN_2);
            assert_eq!(f32x4_reduce_add(values), 7.9672356);
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
        use core::f32::consts::{E, LN_2, PI, SQRT_2};

        use super::f32x4_reduce_add;

        unsafe {
            let values = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
            assert_eq!(f32x4_reduce_add(values), 10.0);

            let values = _mm_set_ps(PI, E, SQRT_2, LN_2);
            assert_eq!(f32x4_reduce_add(values), 7.9672356);
        }
    }
}
