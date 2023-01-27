test:
	cargo test
	RUSTFLAGS="-C target-feature=+sse,+sse3" cargo test
	RUSTFLAGS="-C target-feature=+sse,+sse3,+avx" cargo test
	RUSTFLAGS="-C target-feature=+sse,+sse3,+avx,+avx512f" cargo +nightly test --all-features

doc:
	cargo +nightly rustdoc --open --all-features -- --cfg docsrs