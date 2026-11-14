// lib.rs — Metatron Dynamics, Inc.
// ABR Relational Attention — Crate Root
//
// Grounding: operators.rs V7 (frozen), Kernel V9, DECLARATION.md V5.0 (2026-11-14)
//
// This crate exports only ABR-kernel-sourced mathematics.
// No legacy math (gradient descent, Adam, cross-entropy, Xavier init) is exported
// from this library.  Those components live exclusively in src/bin/, scoped to the
// demo binaries (wo_demo, held_out_benchmark) that use them.

pub mod declaration;
pub mod operators;
pub mod accumulation;
pub mod attention;
pub mod streaming;
pub mod attention_layer;
pub mod multi_head;
pub mod semantic;
pub mod layer_stack;
