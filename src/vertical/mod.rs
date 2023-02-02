use core::ops::*;
use slicesimd_proc::simd_trait;

cfg_if::cfg_if! {
    if #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse"))] {
        mod x86;
    }
}

#[simd_trait]
pub trait VerticalAdd {
    #[inline]
    fn add_assign (&mut self, rhs: &Self) {
        if !self.add_assign_checked(rhs) {
            panic!("Slice sizes don't match: {} v. {}", self.len(), rhs.len())
        }
    }

    #[inline]
    fn add_assign_checked (&mut self, rhs: &Self) -> bool {
        if self.len() != rhs.len() { return false }
        unsafe { self.add_assign_unchecked(rhs) };
        return true
    }
 
    #[inline]
    unsafe fn add_assign_unchecked(&mut self, rhs: &Self) {
        for (x, y) in self.iter_mut().zip(rhs.iter()) {
            x.add_assign(y)
        }
    }
}

#[simd_trait]
pub trait VerticalSub {
    #[inline]
    fn sub_assign (&mut self, rhs: &Self) {
        if !self.sub_assign_checked(rhs) {
            panic!("Slice sizes don't match: {} v. {}", self.len(), rhs.len())
        }
    }

    #[inline]
    fn sub_assign_checked (&mut self, rhs: &Self) -> bool {
        if self.len() != rhs.len() { return false }
        unsafe { self.sub_assign_unchecked(rhs) };
        return true
    }
 
    #[inline]
    unsafe fn sub_assign_unchecked(&mut self, rhs: &Self) {
        for (x, y) in self.iter_mut().zip(rhs.iter()) {
            x.sub_assign(y)
        }
    }
}

#[simd_trait]
pub trait VerticalMul {
    #[inline]
    fn mul_assign (&mut self, rhs: &Self) {
        if !self.mul_assign_checked(rhs) {
            panic!("Slice sizes don't match: {} v. {}", self.len(), rhs.len())
        }
    }

    #[inline]
    fn mul_assign_checked (&mut self, rhs: &Self) -> bool {
        if self.len() != rhs.len() { return false }
        unsafe { self.mul_assign_unchecked(rhs) };
        return true
    }
 
    #[inline]
    unsafe fn mul_assign_unchecked(&mut self, rhs: &Self) {
        for (x, y) in self.iter_mut().zip(rhs.iter()) {
            x.mul_assign(y)
        }
    }
}

#[simd_trait]
pub trait VerticalDiv {
    #[inline]
    fn div_assign (&mut self, rhs: &Self) {
        if !self.div_assign_checked(rhs) {
            panic!("Slice sizes don't match: {} v. {}", self.len(), rhs.len())
        }
    }

    #[inline]
    fn div_assign_checked (&mut self, rhs: &Self) -> bool {
        if self.len() != rhs.len() { return false }
        unsafe { self.div_assign_unchecked(rhs) };
        return true
    }
 
    #[inline]
    unsafe fn div_assign_unchecked(&mut self, rhs: &Self) {
        for (x, y) in self.iter_mut().zip(rhs.iter()) {
            x.div_assign(y)
        }
    }
}