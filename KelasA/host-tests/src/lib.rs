// no_std DISENGAJA: kalau src/tinyml memakai sesuatu di luar `core`
// (mis. f32::exp, Vec, println), crate ini GAGAL compile -- sama seperti firmware.
#![no_std]

#[path = "../../src/tinyml/mod.rs"]
pub mod tinyml;
