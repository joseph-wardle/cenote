// Unit tests for settings resolution: a source in — the host's settings
// map, or a render settings prim's namespaced settings — one wire
// SettingsPatch and a list of complaints out, plus the merge that decides
// which of the two wins. Resolution is a pure function precisely so this
// can be checked without a stage, a server, or a GPU — every clamp the
// delegate owes the server (a rejected change-set is rejected whole) is a
// line here.
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

#include "pxr/base/tf/staticTokens.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/retainedDataSource.h"
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

// The keys as a host authors them — spelled out again rather than shared
// with the resolver, because the strings are the contract under test.
// clang-format off
TF_DEFINE_PRIVATE_TOKENS(_tokens,
    ((samplesPerPixel, "cenote:samplesPerPixel"))
    ((noiseThreshold, "cenote:noiseThreshold"))
    ((maxBounces, "cenote:maxBounces"))
    ((denoise, "cenote:denoise"))
);
// clang-format on

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
    check(resolved.patch.denoise && !*resolved.patch.denoise,
          "and a batch host is handed the estimator's own pixels");
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
    check(!resolved.patch.denoise, "and an unread denoise switch stays unset");
}

/// The budget has two spellings and the cenote one wins: the standard key
/// is in every map by construction, so a host that also authored ours
/// went out of its way to.
void the_cenote_budget_beats_the_standard_one() {
    const HdCenoteResolvedSettings ours =
        HdCenoteResolveSettings(populated({{_tokens->samplesPerPixel, VtValue(7)}}));
    check(ours.patch.spp == 7U, "cenote:samplesPerPixel is read");
    check(ours.warnings.empty(), "and reading it is silent");

    const HdCenoteResolvedSettings standard = HdCenoteResolveSettings(
        populated({{HdRenderSettingsTokens->convergedSamplesPerPixel, VtValue(9)}}));
    check(standard.patch.spp == 9U, "convergedSamplesPerPixel is read when ours is absent");

    const HdCenoteResolvedSettings both = HdCenoteResolveSettings(
        populated({{_tokens->samplesPerPixel, VtValue(7)},
                   {HdRenderSettingsTokens->convergedSamplesPerPixel, VtValue(9)}}));
    check(both.patch.spp == 7U, "ours wins when a host authored both");
}

/// The three states of the threshold, and the two ways of asking for the
/// stop to be switched off.
void the_threshold_keeps_its_three_states() {
    const auto of = [](float value) {
        return threshold(
            HdCenoteResolveSettings(populated({{_tokens->noiseThreshold, VtValue(value)}})).patch);
    };
    check(of(0.02f) == on(0.02f), "a threshold is set");
    check(of(0.0f) == kOff, "zero is the spelling for off");
    check(of(1.0f) == on(1.0f), "1 is in range");

    const HdCenoteResolvedSettings zero =
        HdCenoteResolveSettings(populated({{_tokens->noiseThreshold, VtValue(0.0f)}}));
    check(zero.warnings.empty(), "and switching it off that way is silent");
}

/// The one key with nothing to clamp: a switch is in range by being one.
/// What it still owes is the difference between "off" and "unsaid" —
/// authored false is a decision the shot made and has to reach the server
/// as one, because the session it lands in may have denoising on.
void the_denoise_switch_says_off_out_loud() {
    const auto of = [](const VtValue& value) {
        return HdCenoteResolveSettings(populated({{_tokens->denoise, value}})).patch.denoise;
    };
    const std::optional<bool> on = of(VtValue(true));
    const std::optional<bool> off = of(VtValue(false));
    check(on.value_or(false), "an authored true is read");
    check(off && !*off, "and an authored false is read as false");

    const HdCenoteResolvedSettings unreadable =
        HdCenoteResolveSettings(populated({{_tokens->denoise, VtValue(std::string("yes"))}}));
    check(!unreadable.patch.denoise, "a switch nobody can read stays unset");
    check(unreadable.warnings.size() == 1, "and says so");
    check(unreadable.warnings.front().find("as a switch") != std::string::npos,
          "naming what it wanted to be, not a number");
}

/// Every out-of-range value is pulled into range under a warning, because
/// the server would reject the whole flush instead — every mesh in it
/// included.
void out_of_range_values_are_clamped_not_refused() {
    const auto one = [](const TfToken& key, const VtValue& value) {
        return HdCenoteResolveSettings(populated({{key, value}}));
    };

    const HdCenoteResolvedSettings noSamples = one(_tokens->samplesPerPixel, VtValue(0));
    check(noSamples.patch.spp == 1U, "a budget of zero becomes one sample");
    check(noSamples.warnings.size() == 1, "and says so");

    const HdCenoteResolvedSettings deep = one(_tokens->maxBounces, VtValue(512));
    check(deep.patch.max_bounces == 255U, "512 bounces becomes the renderer's 255");
    check(deep.warnings.size() == 1, "and says so");

    const HdCenoteResolvedSettings flat = one(_tokens->maxBounces, VtValue(0));
    check(flat.patch.max_bounces == 1U, "zero bounces becomes one");
    check(flat.warnings.size() == 1, "and says so");

    const HdCenoteResolvedSettings loose = one(_tokens->noiseThreshold, VtValue(2.0f));
    check(threshold(loose.patch) == on(1.0f), "a threshold above 1 becomes 1");
    check(loose.warnings.size() == 1, "and says so");

    const HdCenoteResolvedSettings negative = one(_tokens->noiseThreshold, VtValue(-1.0f));
    check(threshold(negative.patch) == kOff,
          "a negative threshold is not a relative error, so the stop is off");
    check(negative.warnings.size() == 1, "and says so");
}

/// A value of the wrong type is a complaint and nothing else: guessing
/// would overwrite what the scene said with a number nobody authored.
void an_unreadable_value_is_ignored() {
    const HdCenoteResolvedSettings resolved =
        HdCenoteResolveSettings(populated({{_tokens->maxBounces, VtValue(std::string("deep"))}}));
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

/// A render settings prim's namespacedSettings as usdImaging builds it:
/// flat, keyed by the whole attribute name, one sampled source per value.
HdContainerDataSourceHandle authored(std::initializer_list<std::pair<TfToken, VtValue>> settings) {
    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;
    for (const auto& [key, value] : settings) {
        names.push_back(key);
        values.push_back(HdRetainedSampledDataSource::New(value));
    }
    return HdRetainedContainerDataSource::New(names.size(), names.data(), values.data());
}

/// The prim path reads the same keys through the same clamps — it is
/// literally the same function, and this is the line that says so.
void a_settings_prim_resolves_like_a_map() {
    const HdCenoteResolvedSettings resolved = HdCenoteResolveNamespacedSettings(
        authored({{_tokens->samplesPerPixel, VtValue(12)}, {_tokens->maxBounces, VtValue(999)}}));
    check(resolved.patch.spp == 12U, "a prim's budget is read");
    check(resolved.patch.max_bounces == 255U, "and its depth is clamped like anyone else's");
    check(resolved.warnings.size() == 1, "under one complaint");
    check(!resolved.patch.noise_threshold, "a key the prim does not author stays unset");

    const HdCenoteResolvedSettings none = HdCenoteResolveNamespacedSettings(nullptr);
    check(!none.patch.spp && !none.patch.max_bounces, "a prim with no settings patches nothing");
    check(none.warnings.empty(), "and complains about nothing");
}

/// Precedence, per key: the shot's opinion beats the session's where the
/// shot has one, and leaves the session's standing where it does not.
void the_prim_wins_the_keys_it_authors() {
    const HdCenoteResolvedSettings map =
        HdCenoteResolveSettings(populated({{_tokens->samplesPerPixel, VtValue(64)}}));
    const HdCenoteResolvedSettings prim =
        HdCenoteResolveNamespacedSettings(authored({{_tokens->samplesPerPixel, VtValue(512)}}));

    const HdCenoteResolvedSettings merged = HdCenoteOverlaySettings(map, prim);
    check(merged.patch.spp == 512U, "the prim's budget wins");
    check(merged.patch.max_bounces == 8U, "and the map still answers for what the prim skipped");
    check(threshold(merged.patch) == kOff, "for every key it skipped");

    // The other direction is what happens when the prim goes away: the
    // map is re-resolved whole, so nothing of the prim's survives.
    check(HdCenoteOverlaySettings(map, {}).patch.spp == 64U,
          "and an empty overlay leaves the map alone");
}

/// Complaints from both sources reach the delegate in one list, the map's
/// first — a prim that names a key nobody has does not silence a map that
/// does the same.
void complaints_from_both_sources_survive_the_merge() {
    const HdCenoteResolvedSettings map =
        HdCenoteResolveSettings(populated({{TfToken("cenote:maxBounce"), VtValue(4)}}));
    const HdCenoteResolvedSettings prim =
        HdCenoteResolveNamespacedSettings(authored({{TfToken("cenote:sampels"), VtValue(4)}}));
    const HdCenoteResolvedSettings merged = HdCenoteOverlaySettings(map, prim);
    check(merged.warnings.size() == 2, "both typos are reported");
    check(merged.warnings.front() == map.warnings.front(), "the map's complaint comes first");
}

/// The status line the delegate posts — and the string a batch host with
/// no settings UI is observed through (tests/render_settings_test.py greps
/// exactly these phrases), so its wording is a contract, not a log.
void the_description_names_what_was_resolved() {
    const std::string defaults =
        HdCenoteDescribeSettings(HdCenoteResolveSettings(populated()).patch);
    check(defaults ==
              "cenote: rendering at 64 samples per pixel, no early stop, 8 bounces, not denoised",
          "the defaults describe themselves in full");

    const std::string filtered = HdCenoteDescribeSettings(
        HdCenoteResolveSettings(populated({{_tokens->denoise, VtValue(true)}})).patch);
    check(filtered.ends_with(", denoised"), "and a filtered render says which it is");

    const std::string stopping = HdCenoteDescribeSettings(
        HdCenoteResolveSettings(populated({{_tokens->noiseThreshold, VtValue(0.05f)}})).patch);
    check(stopping.find("stopping early at 0.05 relative error") != std::string::npos,
          "a threshold is named as the relative error it is");

    // Every field unset is the honest case the delegate's own populated
    // map never produces: nothing was decided, so nothing is claimed.
    const std::string nothing = HdCenoteDescribeSettings({});
    check(nothing == "cenote: rendering at the sample budget already in force, the early stop "
                     "already in force, the depth already in force, the denoising already in "
                     "force",
          "an empty patch claims no numbers");
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
    the_denoise_switch_says_off_out_loud();
    out_of_range_values_are_clamped_not_refused();
    an_unreadable_value_is_ignored();
    unknown_cenote_keys_warn();
    a_settings_prim_resolves_like_a_map();
    the_prim_wins_the_keys_it_authors();
    complaints_from_both_sources_survive_the_merge();
    the_description_names_what_was_resolved();

    if (failures > 0) {
        std::println(stderr, "{} failure(s)", failures);
        return 1;
    }
    std::println("all settings tests passed");
    return 0;
}
