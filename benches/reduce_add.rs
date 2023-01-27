#![feature(iterator_try_collect)]

use std::{thread::{available_parallelism}};
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::{thread_rng, Rng, distributions::Uniform};
use rayon::{prelude::{IntoParallelIterator, ParallelIterator}, slice::{ParallelSliceMut}};
use slicesimd::reduce_add_in_place;

pub fn benchmark_reduce_add(c: &mut Criterion) {
    let input = thread_rng().sample_iter(Uniform::new(-100.0, 100.0f32)).take(1_000_000).collect::<Vec<_>>();
    
    c.bench_with_input(BenchmarkId::new("simdslice w/ copy", 0), &input, |b, input| {
        b.iter(|| {
            let mut simdslice = input.clone();
            reduce_add_in_place(&mut simdslice)
        })
    });

    c.bench_with_input(BenchmarkId::new("simdslice w/o copy", 1), &input, |b, input| {
        let mut simdslice = input.clone();
        b.iter(|| reduce_add_in_place(&mut simdslice))
    });

    c.bench_with_input(BenchmarkId::new("naive", 2), &input, |b, input| {
        b.iter(|| input.iter().sum::<f32>())
    });

    c.bench_with_input(BenchmarkId::new("rayon", 3), &input, |b, input| {
        b.iter(|| input.into_par_iter().sum::<f32>())
    });

    c.bench_with_input(BenchmarkId::new("rayon simdslice (w/o copy)", 3), &input, |b, input| {
        let mut simdslice = input.clone();
        let threads = available_parallelism().unwrap().get();
        let len = simdslice.len() / threads;

        b.iter(|| {
            let mut result = simdslice.par_chunks_mut(len).map(reduce_add_in_place).collect::<Vec<_>>();
            return reduce_add_in_place(&mut result)
        })
    });
}

criterion_group!(benches, benchmark_reduce_add);
criterion_main!(benches);