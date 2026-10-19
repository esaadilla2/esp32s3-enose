//! Fungsi matematika minimal untuk `no_std` TANPA dependency tambahan.
//!
//! `f32::exp` / `f64::sqrt` tidak tersedia di `core` pada target bare-metal
//! (butuh libm). Dua fungsi di bawah cukup untuk pipeline ini dan diuji
//! terhadap implementasi `std` di host (lihat host-tests).

/// sqrt(x) untuk f64 via Newton-Raphson. `x < 0`, NaN -> NaN; 0 -> 0.
pub fn sqrt_f64(x: f64) -> f64 {
    if x != x || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 || x == f64::INFINITY {
        return x;
    }
    // Tebakan awal dari manipulasi eksponen, lalu 6 iterasi Newton
    // (kuadratik -> presisi penuh f64).
    let mut g = f64::from_bits((x.to_bits() >> 1) + 0x1FF8_0000_0000_0000);
    let mut i = 0;
    while i < 6 {
        g = 0.5 * (g + x / g);
        i += 1;
    }
    g
}

const LOG2_E: f32 = core::f32::consts::LOG2_E;
// ln2 dipecah (Cody-Waite) supaya r = x - k*ln2 tetap akurat untuk |x| besar.
const LN_2_HI: f32 = 0.693_145_75;
const LN_2_LO: f32 = 1.428_606_8e-6;

/// exp(x) untuk x <= 0 (dipakai softmax setelah dikurangi nilai maksimum).
/// x > 0 dijepit ke 0; x < -87 -> 0.0. Galat relatif ~1e-7.
pub fn exp_nonpositive_f32(x: f32) -> f32 {
    if x != x {
        return f32::NAN;
    }
    let x = if x > 0.0 { 0.0 } else { x };
    if x < -87.0 {
        return 0.0;
    }
    let t = x * LOG2_E;
    let mut k = t as i32; // trunc ke arah 0
    if (k as f32) > t {
        k -= 1; // jadi floor untuk nilai negatif
    }
    let kf = k as f32;
    let r = (x - kf * LN_2_HI) - kf * LN_2_LO; // r di [0, ln2)
    // Taylor derajat 7 untuk exp(r), r < 0.7
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0
                        + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    let pow2 = f32::from_bits(((k + 127) as u32) << 23); // 2^k, k di [-126, 0]
    p * pow2
}
