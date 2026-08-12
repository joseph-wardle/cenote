// Unit tests for settings resolution: the map a host hands us in, one
// wire SettingsPatch and a list of complaints out. Resolution is a pure
// function precisely so this can be checked without a stage, a server, or
// a GPU — every clamp the delegate owes the server (a rejected change-set
// is rejected whole) is a line here.
#include "renderSettings.hpp"

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/tokens.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

using cenote::wire::Set;

int failures = 0;

void fail(std::string_view message) {
    ++failures;
    std::println(stderr, "FAIL: {}", message);
}

void check(bool condition, std::string_view label) {
    if (!condition) {
        fail(label);
    }
}

// The keys as a host authors them, behind accessors for the same reason
// the resolver's are: a TfToken built before Tf's own static registry
// throws where nothing catches.
const TfToken& kSamples() {
    static const TfToken token("cenote:samplesPerPixel");
    return token;
}
const TfToken& kThreshold() {
    static const TfToken token("cenote:noiseThreshold");
    return token;
}
const TfToken& kBounces() {
    static const TfToken token("cenote:maxBounces");
    return token;
}

/// The map every host actually has: whatever it authored, over the
/// advertised defaults the delegate populates.
HdRenderSettingsMap populated(std::initializer_list<std::pair<TfToken, VtValue>> authored = {}) {
    HdRenderSettingsMap map;
    for (const auto& [key, value] : authored) {
        map[key] = value;
    }
    for (const HdRenderSettingDescriptor& descriptor : HdCenoteSettingDescriptors()) {
        if (map.find(descriptor.key) == map.end()) {
            map[descriptor.key] = descriptor.defaultValue;
        }
    }
    return map;
}

/// The threshold's three states, flattened so they compare: no outer
/// value is the patch leaving the field alone, an empty inner one is
/// `Clear` — the early stop switched off.
using Threshold = std::optional<std::optional<float>>;
const Threshold kUnset{};
const Threshold kOff{std::optional<float>{}};
Threshold on(float value) { return Threshold{std::optional<float>{value}}; }

Threshold threshold(const cenote::wire::SettingsPatch& patch) {
    if (!patch.noise_threshold) {
        return kUnset;
    }
    if (const Set<float>* set = std::get_if<Set<float>>(&*patch.noise_threshold)) {
        return on(set->value);
    }
    return kOff;
}

/// An unauthored stage renders the advertised defaults, silently, onto
/// the settings object the server names.
void defaults() {
    const HdCenoteResolvedSettings resolved = HdCenoteResolveSettings(populated());
    check(resolved.warnings.empty(), "the defaults warn about nothing");
    check(resolved.patch.name == "settings", "the patch names the scene's settings object");
    check(resolved.patch.spp == 64U, "the default budget is 64 samples");
    check(threshold(resolved.patch) == kOff, "the default threshold clears the early stop");
    check(resolved.patch.max_bounces == 8U, "the default depth is 8 bounces");
    check(!resolved.patch.resolution, "resolution is not a setting");
    check(!resolved.patch.seed, "seed is not a setting");
}

/// A key the map does not carry leaves its field unset — the wire's
/// "leave this alone" — so nothing here overwrites what a scene said.
void absent_keys_leave_the_field_alone() {
    const HdCenoteResolvedSettings resolved = HdCenoteResolveSettings({});
    check(resolved.warnings.empty(), "an empty map warns about nothing");
    check(!resolved.patch.spp, "an unread budget stays unset");
    check(!resolved.patch.noise_threshold, "an unread threshold stays unset");
    check(!resolved.patch.max_bounces, "an unread depth stays unset");
}

/// The budget has two spellings and the cenote one wins: the standard key
/// is in every map by construction, so a host that also authored ours
/// went out of its way to.
void the_cenote_budget_beats_the_standard_one() {
    const HdCenoteResolvedSettings ours =
        HdCenoteResolveSettings(populated({{kSamples(), VtValue(7)}}));
    check(ours.patch.spp == 7U, "cenote:samplesPerPixel is read");
    check(ours.warnings.empty(), "and reading it is silent");

    const HdCenoteResolvedSettings standard = HdCenoteResolveSettings(
        populated({{HdRenderSettingsTokens->convergedSamplesPerPixel, VtValue(9)}}));
    check(standard.patch.spp == 9U, "convergedSamplesPerPixel is read when ours is absent");

    const HdCenoteResolvedSettings both = HdCenoteResolveSettings(
        populated({{kSamples(), VtValue(7)},
                   {HdRenderSettingsTokens->convergedSamplesPerPixel, VtValue(9)}}));
    check(both.patch.spp == 7U, "ours wins when a host authored both");
}

/// The three states of the threshold, and the two ways of asking for the
/// stop to be switched off.
void the_threshold_keeps_its_three_states() {
    const auto of = [](float value) {
        return threshold(
            HdCenoteResolveSettings(populated({{kThreshold(), VtValue(value)}})).patch);
    };
    check(of(0.02f) == on(0.02f), "a threshold is set");
    check(of(0.0f) == kOff, "zero is the spelling for off");
    check(of(1.0f) == on(1.0f), "1 is in range");

    const HdCenoteResolvedSettings zero =
        HdCenoteResolveSettings(populated({{kThreshold(), VtValue(0.0f)}}));
    check(zero.warnings.empty(), "and switching it off that way is silent");
}

/// Every out-of-range value is pulled into range under a warning, because
/// the server would reject the whole flush instead — every mesh in it
/// included.
void out_of_range_values_are_clamped_not_refused() {
    const auto one = [](const TfToken& key, const VtValue& value) {
        return HdCenoteResolveSettings(populated({{key, value}}));
    };

    const HdCenoteResolvedSettings noSamples = one(kSamples(), VtValue(0));
    check(noSamples.patch.spp == 1U, "a budget of zero becomes one sample");
    check(noSamples.warnings.size() == 1, "and says so");

    const HdCenoteResolvedSettings deep = one(kBounces(), VtValue(512));
    check(deep.patch.max_bounces == 255U, "512 bounces becomes the renderer's 255");
    check(deep.warnings.size() == 1, "and says so");

    const HdCenoteResolvedSettings flat = one(kBounces(), VtValue(0));
    check(flat.patch.max_bounces == 1U, "zero bounces becomes one");
    check(flat.warnings.size() == 1, "and says so");

    const HdCenoteResolvedSettings loose = one(kThreshold(), VtValue(2.0f));
    check(threshold(loose.patch) == on(1.0f), "a threshold above 1 becomes 1");
    check(loose.warnings.size() == 1, "and says so");

    const HdCenoteResolvedSettings negative = one(kThreshold(), VtValue(-1.0f));
    check(threshold(negative.patch) == kOff,
          "a negative threshold is not a relative error, so the stop is off");
    check(negative.warnings.size() == 1, "and says so");
}

/// A value of the wrong type is a complaint and nothing else: guessing
/// would overwrite what the scene said with a number nobody authored.
void an_unreadable_value_is_ignored() {
    const HdCenoteResolvedSettings resolved =
        HdCenoteResolveSettings(populated({{kBounces(), VtValue(std::string("deep"))}}));
    check(!resolved.patch.max_bounces, "an unreadable depth stays unset");
    check(resolved.warnings.size() == 1, "and says so");
}

/// A typo in a `cenote:` key is otherwise perfectly silent — the map
/// would carry it forever and nothing would read it. Keys in anyone
/// else's namespace are not ours to judge.
void unknown_cenote_keys_warn() {
    const HdCenoteResolvedSettings ours =
        HdCenoteResolveSettings(populated({{TfToken("cenote:maxBounce"), VtValue(4)}}));
    check(ours.warnings.size() == 1, "a misspelled cenote key warns");
    check(ours.patch.max_bounces == 8U, "and changes nothing");

    const HdCenoteResolvedSettings theirs = HdCenoteResolveSettings(
        populated({{TfToken("ri:hider:jitter"), VtValue(1)},
                   {HdRenderSettingsTokens->enableSceneMaterials, VtValue(true)}}));
    check(theirs.warnings.empty(), "another renderer's settings pass in silence");
}

} // namespace

// A failed check reports by escaping: terminate prints the what() and the
// exit is nonzero — that is the test contract, not an oversight.
// NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
    defaults();
    absent_keys_leave_the_field_alone();
    the_cenote_budget_beats_the_standard_one();
    the_threshold_keeps_its_three_states();
    out_of_range_values_are_clamped_not_refused();
    an_unreadable_value_is_ignored();
    unknown_cenote_keys_warn();

    if (failures > 0) {
        std::println(stderr, "{} failure(s)", failures);
        return 1;
    }
    std::println("all settings tests passed");
    return 0;
}
