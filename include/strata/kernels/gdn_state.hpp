// include/strata/kernels/gdn_state.hpp - the GDN state's storage width, set once at startup.
//
// The gated delta-net's recurrent state is (S, h_v, S) = 128 x 48 x 128 elements per layer, 36 layers: 113.2 MB
// per sequence in F32. Every decode window reads it twice (record and commit) and every conversation checkpoint
// carries a full copy, so halving the element - IEEE binary16 instead of binary32 - halves all three. The math
// stays F32 inside the kernels: an element is widened on load, narrowed on store, and the recurrence's rounding
// is the only difference from the F32 state (HyperQwen measured the same change on the same layer family:
// perplexity 8.045 against 8.046).
//
// The conv history that follows the recurrence in every layer block stays F32: it is 1.1 MB per sequence and
// every conv kernel would have to grow a second dtype for it.
//
// The flag is read by the kernel launchers at capture time (like `native_gdn_set_enabled`), so a captured graph
// replays the kernels it was captured with; set it before the first session. Off (the default) keeps every
// buffer, byte count and kernel exactly as before.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// Configure once at startup, before any session is created or graph captured. `--gdn-f16` /
/// `STRATA_GDN_F16=1` reach this through the engine's options; tests and tools leave it off.
void gdn_state_set_f16(bool on);
bool gdn_state_f16();

/// One recurrent-state element: 4 bytes off, 2 bytes on.
uint64_t gdn_state_esize();

/// One GDN layer's state block as the session arena lays it out: the recurrent part in the configured
/// width, then the conv history in F32.
uint64_t gdn_state_layer_bytes(int64_t state_size, int64_t v_heads, int64_t conv_channels, int64_t d_conv);

/// The recurrent part alone, in bytes - also the conv history's byte offset inside a layer block.
uint64_t gdn_state_rec_bytes(int64_t state_size, int64_t v_heads);

/// Convert a whole state arena between widths on the host: each of `rows` layer blocks holds `rec` recurrent
/// elements (the converted span) followed by `conv_floats` F32 conv-history floats, which are copied verbatim.
/// `dst_f16` names the DESTINATION's width, independently of the running flag, so both directions are one call.
/// Session files additionally carry a configuration fingerprint and are refused outright when the engine's
/// configuration differs, so a checkpoint written by an F32 run normally never reaches an F16 run; this path is
/// the safety net for any flow that does hand one over (and for the width-agnostic blob validators). The
/// F32->F16 narrowing is round-to-nearest-even with NaN and infinity preserved - the values CUDA's `__float2half` produces.
void gdn_state_convert_arena(void* dst, bool dst_f16, const void* src, size_t rows, size_t rec, size_t conv_floats);

}  // namespace strata::kernels
