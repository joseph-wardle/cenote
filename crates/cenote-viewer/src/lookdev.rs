//! The lookdev panel: a live material inspector wired to the render
//! session's edit channel. It lists the scene's materials by name and, for
//! the selected one, shows its `OpenPBR` constants as widgets; moving a
//! widget produces a change-set the viewer feeds to the session (stop,
//! apply, re-prep, restart) — the same channel a scene-file save rides.
//!
//! This is a deliberately temporary tool for testing closures inside the
//! viewer. Once the Hydra render delegate lands, material lookdev moves to
//! Solaris/Karma and this panel goes away, so it stays thin: constants
//! only (a textured parameter is shown read-only, never overridden),
//! materials only (lights, camera, and environment ride the watched scene
//! file), and no state of its own beyond which material is selected. The
//! material values live in the viewer's scene replica, read fresh each
//! frame — the panel never caches them.

use cenote::scene::changeset::MaterialPatch;
use cenote::scene::description::{Material, SceneDescription, Texturable};

/// The panel's only persistent state: which material the list has
/// selected, when one is and it still exists.
#[derive(Default)]
pub struct Lookdev {
    selected: Option<String>,
}

impl Lookdev {
    /// Draw the panel over `description`'s materials and return the edit a
    /// widget produced this frame, if any: the target material's name and
    /// a patch of its current values. The caller applies it to both the
    /// scene replica and the render session, keeping them in lock-step.
    pub fn show(
        &mut self,
        context: &egui::Context,
        description: &SceneDescription,
    ) -> Option<(String, MaterialPatch)> {
        let mut edit = None;
        egui::Window::new("materials")
            .default_pos([12.0, 190.0])
            .resizable(false)
            .show(context, |ui| edit = self.body(ui, description));
        edit
    }

    /// The window's contents: the material picker, then the selected
    /// material's parameter widgets. Returns the edit, if the widgets
    /// moved a value off what the replica holds.
    fn body(
        &mut self,
        ui: &mut egui::Ui,
        description: &SceneDescription,
    ) -> Option<(String, MaterialPatch)> {
        let materials = description.materials();
        if materials.is_empty() {
            ui.weak("no materials in this scene");
            return None;
        }
        // Reconcile the selection against the live set: a reload may have
        // dropped the selected material, and a fresh scene starts with none.
        if self
            .selected
            .as_deref()
            .is_some_and(|name| !materials.contains_key(name))
        {
            self.selected = None;
        }
        if self.selected.is_none() {
            self.selected = materials.keys().next().cloned();
        }
        egui::ComboBox::from_label("material")
            .selected_text(self.selected.as_deref().unwrap_or_default())
            .show_ui(ui, |ui| {
                for name in materials.keys() {
                    ui.selectable_value(&mut self.selected, Some(name.clone()), name);
                }
            });

        let name = self.selected.clone().expect("reconciled to Some above");
        let before = &materials[&name];
        let mut after = before.clone();
        ui.separator();
        egui::ScrollArea::vertical()
            .max_height(360.0)
            .show(ui, |ui| widgets(ui, &mut after));
        (after != *before).then(|| {
            let patch = full_patch(&name, &after);
            (name, patch)
        })
    }
}

/// Every constant parameter, grouped as `OpenPBR` groups them. Textured
/// slots (the five [`Texturable`] fields, plus the normal map) show
/// read-only — this panel edits constants, not maps.
fn widgets(ui: &mut egui::Ui, m: &mut Material) {
    egui::CollapsingHeader::new("Base")
        .default_open(true)
        .show(ui, |ui| {
            texturable_color(ui, "color", &mut m.base_color);
            slider(
                ui,
                "diffuse roughness",
                &mut m.base_diffuse_roughness,
                0.0..=1.0,
            );
            texturable_scalar(ui, "metalness", &mut m.base_metalness, 0.0..=1.0);
        });
    ui.collapsing("Specular", |ui| {
        slider(ui, "weight", &mut m.specular_weight, 0.0..=1.0);
        texturable_scalar(ui, "roughness", &mut m.specular_roughness, 0.0..=1.0);
        slider(ui, "ior", &mut m.specular_ior, 1.0..=3.0);
    });
    ui.collapsing("Transmission", |ui| {
        slider(ui, "weight", &mut m.transmission_weight, 0.0..=1.0);
        color(ui, "color", &mut m.transmission_color);
        amount(ui, "depth", &mut m.transmission_depth);
    });
    ui.collapsing("Coat", |ui| {
        slider(ui, "weight", &mut m.coat_weight, 0.0..=1.0);
        color(ui, "color", &mut m.coat_color);
        slider(ui, "roughness", &mut m.coat_roughness, 0.0..=1.0);
        slider(ui, "ior", &mut m.coat_ior, 1.0..=3.0);
        slider(ui, "darkening", &mut m.coat_darkening, 0.0..=1.0);
    });
    ui.collapsing("Fuzz", |ui| {
        slider(ui, "weight", &mut m.fuzz_weight, 0.0..=1.0);
        color(ui, "color", &mut m.fuzz_color);
        slider(ui, "roughness", &mut m.fuzz_roughness, 0.0..=1.0);
    });
    ui.collapsing("Emission", |ui| {
        amount(ui, "luminance", &mut m.emission_luminance);
        texturable_color(ui, "color", &mut m.emission_color);
    });
    ui.collapsing("Geometry", |ui| {
        texturable_scalar(ui, "opacity", &mut m.geometry_opacity, 0.0..=1.0);
        ui.checkbox(&mut m.geometry_thin_walled, "thin walled");
        if m.geometry_normal.is_some() {
            ui.weak("normal  (textured)");
        }
    });
}

/// A bounded scalar slider.
fn slider(ui: &mut egui::Ui, label: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>) {
    ui.add(egui::Slider::new(value, range).text(label));
}

/// An unbounded non-negative amount (luminance, transmission depth) — no
/// natural upper limit, so a drag box rather than a slider.
fn amount(ui: &mut egui::Ui, label: &str, value: &mut f32) {
    ui.add(
        egui::DragValue::new(value)
            .range(0.0..=f32::INFINITY)
            .speed(0.05)
            .prefix(format!("{label}  ")),
    );
}

/// A linear-RGB color square. The value is linear `Rec.709`, which is what
/// egui's rgb picker edits; the swatch is gamma-correct.
fn color(ui: &mut egui::Ui, label: &str, rgb: &mut [f32; 3]) {
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(rgb);
        ui.label(label);
    });
}

/// A [`Texturable`] scalar: its constant is a slider; a textured binding is
/// read-only.
fn texturable_scalar(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut Texturable<f32>,
    range: std::ops::RangeInclusive<f32>,
) {
    match value {
        Texturable::Constant(value) => slider(ui, label, value, range),
        Texturable::Texture(_) => {
            ui.weak(format!("{label}  (textured)"));
        }
    }
}

/// A [`Texturable`] color: its constant is a color square; a textured
/// binding is read-only.
fn texturable_color(ui: &mut egui::Ui, label: &str, value: &mut Texturable<[f32; 3]>) {
    match value {
        Texturable::Constant(rgb) => color(ui, label, rgb),
        Texturable::Texture(_) => {
            ui.weak(format!("{label}  (textured)"));
        }
    }
}

/// A patch carrying every field of `material` — the whole surface, not a
/// minimal diff. The apply path's equality gate makes this exact: only the
/// field the widget actually moved differs from what both replicas hold, so
/// only it dirties and forces a re-prep. Sending the whole material keeps
/// the panel free of per-field change tracking.
fn full_patch(name: &str, material: &Material) -> MaterialPatch {
    MaterialPatch {
        base_color: Some(material.base_color.clone()),
        base_diffuse_roughness: Some(material.base_diffuse_roughness),
        base_metalness: Some(material.base_metalness.clone()),
        specular_weight: Some(material.specular_weight),
        specular_roughness: Some(material.specular_roughness.clone()),
        specular_ior: Some(material.specular_ior),
        transmission_weight: Some(material.transmission_weight),
        transmission_color: Some(material.transmission_color),
        transmission_depth: Some(material.transmission_depth),
        coat_weight: Some(material.coat_weight),
        coat_color: Some(material.coat_color),
        coat_roughness: Some(material.coat_roughness),
        coat_ior: Some(material.coat_ior),
        coat_darkening: Some(material.coat_darkening),
        fuzz_weight: Some(material.fuzz_weight),
        fuzz_color: Some(material.fuzz_color),
        fuzz_roughness: Some(material.fuzz_roughness),
        emission_luminance: Some(material.emission_luminance),
        emission_color: Some(material.emission_color.clone()),
        geometry_opacity: Some(material.geometry_opacity.clone()),
        geometry_thin_walled: Some(material.geometry_thin_walled),
        geometry_normal: Some(material.geometry_normal.clone()),
        ..MaterialPatch::new(name)
    }
}
