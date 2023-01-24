#[inline]
pub fn reduce_add (v: &mut [f32]) -> f32 {
    return v.into_iter().sum()
}