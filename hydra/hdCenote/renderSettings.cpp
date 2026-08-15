#include "renderSettings.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>
#include <variant>

#include "pxr/base/tf/staticTokens.h"
#include "pxr/base/tf/stringUtils.h"
#include "pxr/base/tf/token.h"
#include "pxr/imaging/hd/tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

// The keys as a host authors them.
// clang-format off
TF_DEFINE_PRIVATE_TOKENS(_tokens,
    ((samplesPerPixel, "cenote:samplesPerPixel"))
    ((noiseThreshold, "cenote:noiseThreshold"))
    ((maxBounces, "cenote:maxBounces"))
    ((denoise, "cenote:denoise"))
);
// clang-format on

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
// Denoising is off, unlike the viewer's own default: a host asks this
// delegate for an image it will write to disk or composite, and the
// estimator's own pixels are the honest answer to that until somebody
// says otherwise.
constexpr int kDefaultSamplesPerPixel = 64;
constexpr float kDefaultNoiseThreshold = 0.0f;
constexpr int kDefaultMaxBounces = 8;
constexpr bool kDefaultDenoise = false;

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
/// can parse. The complaint names what the key wanted to be — the type
/// says it, so no call site has to.
template <typename T>
std::optional<T> _Value(const HdRenderSettingsMap& settings, const TfToken& key,
                        std::vector<std::string>* warnings) {
    const auto entry = settings.find(key);
    if (entry == settings.end()) {
        return std::nullopt;
    }
    const VtValue value = VtValue::Cast<T>(entry->second);
    if (value.IsEmpty()) {
        constexpr const char* wanted = std::is_same_v<T, bool> ? "switch" : "number";
        warnings->push_back(TfStringPrintf("%s is a %s, which cenote cannot read as a %s; "
                                           "the setting is ignored",
                                           key.GetText(), entry->second.GetTypeName().c_str(),
                                           wanted));
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
                                           _tokens->noiseThreshold.GetText(), value));
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
                                           _tokens->noiseThreshold.GetText(), value));
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
        if (key == _tokens->samplesPerPixel || key == _tokens->noiseThreshold ||
            key == _tokens->maxBounces || key == _tokens->denoise) {
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
        {"Noise Threshold", _tokens->noiseThreshold, VtValue(kDefaultNoiseThreshold)},
        {"Max Bounces", _tokens->maxBounces, VtValue(kDefaultMaxBounces)},
        {"Denoise", _tokens->denoise, VtValue(kDefaultDenoise)},
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
    TfToken budgetKey = _tokens->samplesPerPixel;
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
            _Value<float>(settings, _tokens->noiseThreshold, warnings)) {
        _Threshold(*threshold, &resolved.patch, warnings);
    }

    if (const std::optional<int> bounces = _Value<int>(settings, _tokens->maxBounces, warnings)) {
        resolved.patch.max_bounces = static_cast<std::uint32_t>(
            _Clamped(*bounces, 1, kMaxBounces, _tokens->maxBounces, warnings));
    }

    // Nothing to clamp: every bool is in range, and the only reading that
    // can go wrong is a value that is not one — which `_Value` already
    // complains about and ignores.
    if (const std::optional<bool> denoise = _Value<bool>(settings, _tokens->denoise, warnings)) {
        resolved.patch.denoise = denoise;
    }

    _Unknown(settings, warnings);
    return resolved;
}

HdCenoteResolvedSettings
HdCenoteResolveNamespacedSettings(const HdContainerDataSourceHandle& namespacedSettings) {
    HdRenderSettingsMap settings;
    if (namespacedSettings) {
        for (const TfToken& name : namespacedSettings->GetNames()) {
            // Time zero: a render setting is a uniform, and a scene that
            // animates one is asking for something the wire has no way to
            // say.
            if (const auto source = HdSampledDataSource::Cast(namespacedSettings->Get(name))) {
                settings[name] = source->GetValue(0.0f);
            }
        }
    }
    return HdCenoteResolveSettings(settings);
}

std::string HdCenoteDescribeSettings(const cenote::wire::SettingsPatch& patch) {
    std::string description = "cenote: rendering at ";
    description += patch.spp ? TfStringPrintf("%u samples per pixel", *patch.spp)
                             : std::string("the sample budget already in force");
    if (!patch.noise_threshold) {
        description += ", the early stop already in force";
    } else if (const auto* stop = std::get_if<cenote::wire::Set<float>>(&*patch.noise_threshold)) {
        description += TfStringPrintf(", stopping early at %g relative error", stop->value);
    } else {
        description += ", no early stop";
    }
    description += patch.max_bounces ? TfStringPrintf(", %u bounces", *patch.max_bounces)
                                     : std::string(", the depth already in force");
    if (!patch.denoise) {
        description += ", the denoising already in force";
    } else {
        description += *patch.denoise ? ", denoised" : ", not denoised";
    }
    return description;
}

HdCenoteResolvedSettings HdCenoteOverlaySettings(HdCenoteResolvedSettings under,
                                                 const HdCenoteResolvedSettings& over) {
    if (over.patch.spp) {
        under.patch.spp = over.patch.spp;
    }
    if (over.patch.noise_threshold) {
        under.patch.noise_threshold = over.patch.noise_threshold;
    }
    if (over.patch.max_bounces) {
        under.patch.max_bounces = over.patch.max_bounces;
    }
    if (over.patch.denoise) {
        under.patch.denoise = over.patch.denoise;
    }
    if (over.patch.seed) {
        under.patch.seed = over.patch.seed;
    }
    under.warnings.insert(under.warnings.end(), over.warnings.begin(), over.warnings.end());
    return under;
}

PXR_NAMESPACE_CLOSE_SCOPE
