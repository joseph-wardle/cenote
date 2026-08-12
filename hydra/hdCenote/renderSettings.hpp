// The render-settings surface: which knobs cenote exports to a host, and
// the resolution that turns a host's settings map into the one wire
// SettingsPatch the server obeys.
//
// Three keys, flat `cenote:` namespace, camelCase — the smallest surface
// that describes render *intent* rather than implementation: the sample
// budget (advertised under Hydra's own `convergedSamplesPerPixel`, and
// also read as `cenote:samplesPerPixel` for hosts that spell it our way),
// `cenote:noiseThreshold`, and `cenote:maxBounces`. Resolution is
// deliberately not among them — it already arrives as the render buffer's
// allocation (renderBuffer.cpp), and a second authority for the same number
// could only disagree with the first.
//
// Resolution is a pure function so it can be tested without a stage, and
// so the prim path can feed the same arithmetic later. It never posts
// diagnostics itself: it hands back the complaints and the delegate posts
// them, which is also what lets the delegate stay quiet about a complaint
// it has already made.
//
// **The delegate clamps, and must.** Server-side a change-set is rejected
// whole and `max_bounces` is eight bits, so an authored 512 would take
// every mesh in the same flush down with it. Every out-of-range value is
// therefore pulled into range here, under a warning — warn and continue,
// never refuse the frame.
#pragma once

#include <string>
#include <vector>

#include "pxr/imaging/hd/renderDelegate.h"
#include "pxr/pxr.h"

#include "wire/scene.hpp"

PXR_NAMESPACE_OPEN_SCOPE

/// What cenote exports to the host's settings UI, and the defaults the
/// base class populates a map with. They match the renderer's own
/// defaults: changing one here changes what an unauthored stage renders.
HdRenderSettingDescriptorList HdCenoteSettingDescriptors();

/// A settings map read as one patch, plus whatever reading it had to
/// complain about — one human-readable sentence per complaint, in the
/// order they should be posted. Built with `TfStringPrintf`, not
/// `std::format`: this compiles into the plugin `.so`, where the
/// portability rule (../README.md) forbids library facilities that need
/// libstdc++ runtime symbols newer than the host's — and `std::format`'s
/// float path wants `std::to_chars` at `GLIBCXX_3.4.31`.
struct HdCenoteResolvedSettings {
    cenote::wire::SettingsPatch patch;
    std::vector<std::string> warnings;
};

/// Resolves `settings` onto the scene's one settings object. A key the
/// map does not carry leaves its field unset, which is the wire's "leave
/// this alone" — so a partial map patches only what it names.
HdCenoteResolvedSettings HdCenoteResolveSettings(const HdRenderSettingsMap& settings);

PXR_NAMESPACE_CLOSE_SCOPE
