#include "renderSettings.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

#include "pxr/base/tf/stringUtils.h"
#include "pxr/base/tf/token.h"
#include "pxr/imaging/hd/tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

/// The keys as a host authors them. Behind an accessor rather than at
/// namespace scope: a TfToken registers itself in Tf's own static
/// registry, and a plugin has no say in which of the two is constructed
/// first.
struct _Keys {
    TfToken samplesPerPixel{"cenote:samplesPerPixel"};
    TfToken noiseThreshold{"cenote:noiseThreshold"};
    TfToken maxBounces{"cenote:maxBounces"};
};

const _Keys& _keys() {
    static const _Keys keys;
    return keys;
}

/// The scene's one settings object, under the name the server's own
/// fallback gives it (`ensure_singletons`, cenote-server/src/main.rs).
/// The name is not load-bearing — the server takes whichever single
/// settings object a scene has — but agreeing on it keeps a delegate
/// patch and a server default from ever reading as two objects.
constexpr const char* kSettingsName = "settings";

/// The prefix that marks a key as addressed to cenote. Anything else in
/// the map belongs to Hydra or to another renderer, and is not ours to
/// complain about.
constexpr std::string_view kNamespace = "cenote:";

// The defaults, which are the renderer's defaults. 64 samples is a
// lookdev budget, not a beauty one: hosts that want to accumulate past it
// say so. 8 bounces is what every corpus baseline is pinned at. A zero
// threshold is the early stop switched off — spend the whole budget.
constexpr int kDefaultSamplesPerPixel = 64;
constexpr float kDefaultNoiseThreshold = 0.0f;
constexpr int kDefaultMaxBounces = 8;

/// The renderer's hard ceiling on path depth
/// (`Wavefront::MAX_BOUNCES_LIMIT`): server-side the field is eight bits
/// and validation rejects the change-set — the whole change-set — above
/// it.
constexpr int kMaxBounces = 255;

/// Reads `key` as a `T`, or nothing when the map does not carry it.
///
/// A value of an uncastable type warns and yields nothing rather than a
/// guess: leaving the field unset leaves the renderer's own answer
/// standing, which is the least destructive reading of a setting no one
/// can parse.
template <typename T>
std::optional<T> _Value(const HdRenderSettingsMap& settings, const TfToken& key,
                        std::vector<std::string>* warnings) {
    const auto entry = settings.find(key);
    if (entry == settings.end()) {
        return std::nullopt;
    }
    const VtValue value = VtValue::Cast<T>(entry->second);
    if (value.IsEmpty()) {
        warnings->push_back(TfStringPrintf("%s is a %s, which cenote cannot read as a number; "
                                           "the setting is ignored",
                                           key.GetText(), entry->second.GetTypeName().c_str()));
        return std::nullopt;
    }
    return value.UncheckedGet<T>();
}

/// `value` pulled into [`low`, `high`], warning when it moved.
int _Clamped(int value, int low, int high, const TfToken& key, std::vector<std::string>* warnings) {
    if (value < low) {
        warnings->push_back(TfStringPrintf("%s is %d; the least cenote renders is %d, using that",
                                           key.GetText(), value, low));
        return low;
    }
    if (value > high) {
        warnings->push_back(TfStringPrintf("%s is %d; the most cenote renders is %d, using that",
                                           key.GetText(), value, high));
        return high;
    }
    return value;
}

/// The threshold as the wire's three states. Zero is the documented
/// spelling for "off" and passes silently; a negative or NaN one is not a
/// relative error at all, and switches the early stop off under a warning.
void _Threshold(float value, cenote::wire::SettingsPatch* patch,
                std::vector<std::string>* warnings) {
    if (std::isnan(value) || value < 0.0f) {
        warnings->push_back(TfStringPrintf("%s is %g, which is not a relative error; "
                                           "the early stop is off",
                                           _keys().noiseThreshold.GetText(), value));
        patch->noise_threshold = cenote::wire::Clear{};
        return;
    }
    if (value == 0.0f) {
        patch->noise_threshold = cenote::wire::Clear{};
        return;
    }
    if (value > 1.0f) {
        warnings->push_back(TfStringPrintf("%s is %g; a relative error above 1 stops on the "
                                           "first sample, using 1",
                                           _keys().noiseThreshold.GetText(), value));
        patch->noise_threshold = cenote::wire::Set{1.0f};
        return;
    }
    patch->noise_threshold = cenote::wire::Set{value};
}

/// Warns about every `cenote:` key that is not one of ours — a typo in a
/// setting is otherwise perfectly silent. Sorted, because the map is a
/// hash and the order a host reads its warnings in should not be.
void _Unknown(const HdRenderSettingsMap& settings, std::vector<std::string>* warnings) {
    std::vector<std::string> unknown;
    for (const auto& [key, value] : settings) {
        const std::string& name = key.GetString();
        if (!std::string_view(name).starts_with(kNamespace)) {
            continue;
        }
        if (key == _keys().samplesPerPixel || key == _keys().noiseThreshold ||
            key == _keys().maxBounces) {
            continue;
        }
        unknown.push_back(name);
    }
    std::sort(unknown.begin(), unknown.end());
    for (const std::string& name : unknown) {
        warnings->push_back(
            TfStringPrintf("%s is not a cenote render setting; it is ignored", name.c_str()));
    }
}

} // namespace

HdRenderSettingDescriptorList HdCenoteSettingDescriptors() {
    // The budget is advertised under Hydra's own token, not ours: it is
    // the one of the three that every host already has a name for, and a
    // host setting the standard key should not have to learn a second.
    return {
        {"Samples Per Pixel", HdRenderSettingsTokens->convergedSamplesPerPixel,
         VtValue(kDefaultSamplesPerPixel)},
        {"Noise Threshold", _keys().noiseThreshold, VtValue(kDefaultNoiseThreshold)},
        {"Max Bounces", _keys().maxBounces, VtValue(kDefaultMaxBounces)},
    };
}

HdCenoteResolvedSettings HdCenoteResolveSettings(const HdRenderSettingsMap& settings) {
    HdCenoteResolvedSettings resolved;
    resolved.patch.name = kSettingsName;
    std::vector<std::string>* const warnings = &resolved.warnings;

    // Ours before Hydra's: the descriptor advertises
    // `convergedSamplesPerPixel`, so populating the defaults puts it in
    // every map — a host that *also* spelled the budget our way meant the
    // one it went out of its way to author.
    TfToken budgetKey = _keys().samplesPerPixel;
    std::optional<int> budget = _Value<int>(settings, budgetKey, warnings);
    if (!budget) {
        budgetKey = HdRenderSettingsTokens->convergedSamplesPerPixel;
        budget = _Value<int>(settings, budgetKey, warnings);
    }
    if (budget) {
        // No ceiling of our own: the budget is a u32 on the wire, and a
        // host asking for more samples than it will ever sit through is
        // asking for a render that never converges, which is its right.
        resolved.patch.spp = static_cast<std::uint32_t>(
            _Clamped(*budget, 1, std::numeric_limits<int>::max(), budgetKey, warnings));
    }

    if (const std::optional<float> threshold =
            _Value<float>(settings, _keys().noiseThreshold, warnings)) {
        _Threshold(*threshold, &resolved.patch, warnings);
    }

    if (const std::optional<int> bounces = _Value<int>(settings, _keys().maxBounces, warnings)) {
        resolved.patch.max_bounces = static_cast<std::uint32_t>(
            _Clamped(*bounces, 1, kMaxBounces, _keys().maxBounces, warnings));
    }

    _Unknown(settings, warnings);
    return resolved;
}

PXR_NAMESPACE_CLOSE_SCOPE
