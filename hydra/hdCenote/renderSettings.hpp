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
// Two sources author those keys and both land here: the host's settings
// map (usdview's sliders, husk's `--settings`) and the stage's active
// UsdRenderSettings prim, which the prim translator (settingsPrim.hpp)
// reads as a `namespacedSettings` container. Same keys, same clamps, one
// arithmetic — they differ only in precedence, and that is one function
// below. The prim wins, because a prim is the shot's own opinion and a
// map is the session's.
//
// Resolution is a pure function so it can be tested without a stage, and
// so both sources feed the same arithmetic. It never posts diagnostics
// itself: it hands back the complaints and the delegate posts them, which
// is also what lets the delegate stay quiet about a complaint it has
// already made.
//
// **The delegate clamps, and must.** Server-side a change-set is rejected
// whole and `max_bounces` is eight bits, so an authored 512 would take
// every mesh in the same flush down with it. Every out-of-range value is
// therefore pulled into range here, under a warning — warn and continue,
// never refuse the frame.
#pragma once

#include <string>
#include <vector>

#include "pxr/imaging/hd/dataSource.h"
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

/// The same resolution over a render settings prim's `namespacedSettings`
/// — a flat container keyed by the whole attribute name
/// (`cenote:maxBounces`), which is how a settings map is keyed too. So the
/// two sources meet at the first line and there is no second set of
/// clamps to drift from this one. A null container is an empty map, which
/// patches nothing.
HdCenoteResolvedSettings
HdCenoteResolveNamespacedSettings(const HdContainerDataSourceHandle& namespacedSettings);

/// The resolved patch as the one line a renderer owes the person waiting
/// on it: what budget, what early stop, what depth this render is actually
/// obeying, after every source and clamp has had its say. Every production
/// renderer prints its resolved settings — it is the first thing anyone
/// asks when a render takes an hour — and it is also the only way a batch
/// host with no settings UI can be *observed* to have delivered them
/// (tests/render_settings_test.py reads exactly this line).
///
/// A field the patch leaves unset says so rather than inventing a number:
/// unset means "leave the renderer's own answer alone", which is not the
/// same claim as any particular value.
std::string HdCenoteDescribeSettings(const cenote::wire::SettingsPatch& patch);

/// `over` laid onto `under`, field by field: a field `over` left unset
/// keeps `under`'s answer, and the complaints concatenate with `under`'s
/// first. This one function is the whole precedence rule between the two
/// sources — per key, never wholesale, so a prim that names only the
/// bounce limit still leaves the map's budget standing. Every field of the
/// patch overlays, not only the three the surface currently authors: the
/// rule belongs to the patch, and a fourth key should not have to
/// remember to come back here.
HdCenoteResolvedSettings HdCenoteOverlaySettings(HdCenoteResolvedSettings under,
                                                 const HdCenoteResolvedSettings& over);

PXR_NAMESPACE_CLOSE_SCOPE
