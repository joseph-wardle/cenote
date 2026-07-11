//! Stable light identity: a per-emissive-instance id that outlives the
//! volatile GPU light index, so a reservoir's stored light reference stays
//! meaningful across scene edits.
//!
//! The GPU light index today is name-order-derived *and* power-filtered in
//! `scene/lower.rs`, so adding or deleting any light two objects away renumbers
//! every light after it — a reservoir storing that index would silently point
//! at the wrong light after the edit. This registry mints a stable id that
//! never moves and never gets reused, and hands back the bidirectional remap
//! the GPU stages translate through: the stable id at rest in the reservoir,
//! the current dense index at the moment of reuse.
//!
//! This substrate lands before its consumer by deliberate plan sequencing (M3
//! §4 step 2 before step 3): the first caller is `restir_candidates`, which
//! stamps a fresh reservoir with the stable id and, on reuse, resolves it back.
//! Until that stage exists the registry is exercised only by its tests, so the
//! module carries one scoped `dead_code` allowance — removed the moment step 3
//! wires the registry into `Scene`'s prep path.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

/// The reserved stable ids. The environment (HDRI) is a reservoir candidate
/// too but is not an instance, so it carries a fixed sentinel; instance ids
/// begin above the reserved block. `LIGHT_ID_NONE` marks an empty or dropped
/// reservoir and doubles as the "no such instance" value in the id→index
/// table — both are `u32::MAX`, matching `LIGHT_NONE` in `lights.slang`.
pub const LIGHT_ID_ENVIRONMENT: u32 = 0;
pub const LIGHT_ID_NONE: u32 = u32::MAX;
/// The first id handed to an emissive instance — past the reserved sentinels.
const FIRST_INSTANCE_ID: u32 = 1;

/// One emissive light present in the current build, as the registry needs to
/// see it: which instance it is, and the fingerprint of everything a stored
/// reservoir sample depends on. Change the fingerprint and the reservoir's
/// `(primitive, barycentrics)` reference is no longer valid, so the history is
/// dropped and a fresh id minted; leave it and the id (and the reuse history)
/// survives an edit to anything else — material, transform, emission.
#[derive(Clone, Debug)]
pub struct EmissiveLight {
    /// The instance's stable source name — the identity key.
    pub name: String,
    /// What the reservoir's triangle reference is relative to: the mesh today
    /// (a mesh edit reshuffles triangle indices). A content marker joins it
    /// once meshes gain in-place geometry edits.
    pub fingerprint: String,
    /// The current dense GPU identity — the TLAS custom index this build gave
    /// the instance. Volatile across edits; that volatility is the whole
    /// reason the registry exists.
    pub instance: u32,
}

/// A live light's registry entry: its stable id and the fingerprint it was
/// last seen with.
#[derive(Clone, Debug)]
struct LiveLight {
    id: u32,
    fingerprint: String,
}

/// The bidirectional remap for one build, the form the GPU stages translate
/// through. `instance_to_id` stamps a fresh candidate with its stable id;
/// `id_to_instance` resolves a reused reservoir's stable id back to the
/// current instance, or to `LIGHT_NONE` when the light is gone (the signal to
/// drop the history).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LightRemap {
    /// Stable id → current TLAS custom index, or `LIGHT_NONE` if the light was
    /// dropped. Indexed by id; length is every id ever minted, so retired ids
    /// keep their slot (reading `LIGHT_NONE`) rather than shifting the rest.
    pub id_to_instance: Vec<u32>,
    /// TLAS custom index → stable id, or `LIGHT_ID_NONE` for a non-emissive
    /// instance. Indexed by custom index; length is the build's instance count.
    pub instance_to_id: Vec<u32>,
}

/// The monotonic, tombstoned light-identity registry. Persists across scene
/// edits (it is scene-level state, reconciled every build), which is what lets
/// a reservoir's stored id stay meaningful while the GPU light table churns
/// underneath it.
#[derive(Default)]
pub struct LightIdentityRegistry {
    /// Emissive-instance name → its stable id and last-seen fingerprint.
    live: BTreeMap<String, LiveLight>,
    /// The next id to mint. Only ever increases: a retired id is never
    /// reissued, so a reservoir referencing a since-deleted light resolves to
    /// `LIGHT_NONE` and is dropped rather than silently retargeted at whatever
    /// took its place. The gap the retired id leaves *is* the tombstone.
    next_id: u32,
}

impl LightIdentityRegistry {
    /// An empty registry — no lights seen yet.
    pub fn new() -> Self {
        Self {
            live: BTreeMap::new(),
            next_id: FIRST_INSTANCE_ID,
        }
    }

    /// Reconcile against the emissive lights of a fresh build and return the
    /// remap. A light keeps its id (and its reuse history) when its name and
    /// fingerprint both match a live entry; it is minted a new id when its
    /// name is new *or* its fingerprint changed (a mesh edit); a live entry
    /// absent from `lights` is dropped, its id retired for good.
    pub fn reconcile(&mut self, lights: &[EmissiveLight], instance_count: u32) -> LightRemap {
        // Drop lights that vanished this build; their ids are never reissued.
        let present: BTreeSet<&str> = lights.iter().map(|light| light.name.as_str()).collect();
        self.live.retain(|name, _| present.contains(name.as_str()));

        let mut instance_to_id = vec![LIGHT_ID_NONE; instance_count as usize];
        for light in lights {
            let id = match self.live.get(&light.name) {
                // Same light, unchanged geometry: keep the id and the history.
                Some(entry) if entry.fingerprint == light.fingerprint => entry.id,
                // New light, or its geometry changed under it: mint fresh,
                // which orphans (tombstones) any prior id for this name.
                _ => {
                    let id = self.next_id;
                    self.next_id += 1;
                    self.live.insert(
                        light.name.clone(),
                        LiveLight {
                            id,
                            fingerprint: light.fingerprint.clone(),
                        },
                    );
                    id
                }
            };
            instance_to_id[light.instance as usize] = id;
        }

        // Size the id→instance table to every id ever minted, so a retired or
        // never-present id reads LIGHT_NONE at its own slot.
        let mut id_to_instance = vec![LIGHT_ID_NONE; self.next_id as usize];
        for light in lights {
            id_to_instance[self.live[&light.name].id as usize] = light.instance;
        }
        LightRemap {
            id_to_instance,
            instance_to_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EmissiveLight, LIGHT_ID_ENVIRONMENT, LIGHT_ID_NONE, LightIdentityRegistry};

    /// Terse `EmissiveLight` for the registry tests.
    fn light(name: &str, fingerprint: &str, instance: u32) -> EmissiveLight {
        EmissiveLight {
            name: name.into(),
            fingerprint: fingerprint.into(),
            instance,
        }
    }

    /// T2: ids stay put while the GPU light index moves under them. Reordering
    /// the lights (an add elsewhere in the scene renumbers every later custom
    /// index) and touching everything but geometry must leave each light's id
    /// — and so its reuse history — exactly where it was. Non-emissive
    /// instances read the "no light" sentinel, and the two remap tables are
    /// inverse over the live lights.
    #[test]
    fn light_ids_are_stable_while_the_gpu_index_churns() {
        let mut registry = LightIdentityRegistry::new();

        // First build: two emitters at custom indices 1 and 3, others dark.
        let first = registry.reconcile(&[light("key", "quad", 1), light("fill", "disc", 3)], 5);
        // Ids are minted monotonically past the reserved environment sentinel.
        assert_ne!(first.instance_to_id[1], LIGHT_ID_ENVIRONMENT);
        assert_eq!(first.instance_to_id[1], 1, "first emitter takes id 1");
        assert_eq!(first.instance_to_id[3], 2, "second emitter takes id 2");
        // The dark instances carry no light id.
        for dark in [0, 2, 4] {
            assert_eq!(first.instance_to_id[dark], LIGHT_ID_NONE);
        }
        // The reserved environment slot in the id table is never an instance.
        assert_eq!(
            first.id_to_instance[LIGHT_ID_ENVIRONMENT as usize],
            LIGHT_ID_NONE
        );

        // Second build: same lights, everything renumbered (custom indices
        // 4 and 0 now), a material edit elsewhere — the ids must not move.
        let second = registry.reconcile(&[light("key", "quad", 4), light("fill", "disc", 0)], 6);
        assert_eq!(
            second.instance_to_id[4], 1,
            "key keeps id 1 across the churn"
        );
        assert_eq!(
            second.instance_to_id[0], 2,
            "fill keeps id 2 across the churn"
        );

        // The tables are inverse over every live light.
        for (instance, &id) in second.instance_to_id.iter().enumerate() {
            if id != LIGHT_ID_NONE {
                assert_eq!(second.id_to_instance[id as usize], instance as u32);
            }
        }
    }

    /// T2: a deleted light retires its id for good — a re-added light of the
    /// same name gets a *fresh* id, never the old one, so a reservoir that
    /// outlived the delete resolves to `LIGHT_NONE` (drop) instead of silently
    /// pointing at the newcomer.
    #[test]
    fn a_deleted_light_retires_its_id_for_good() {
        let mut registry = LightIdentityRegistry::new();
        registry.reconcile(&[light("a", "m", 0), light("b", "m", 1)], 2);

        // Delete 'a'. Its id (1) must now resolve to nothing.
        let dropped = registry.reconcile(&[light("b", "m", 0)], 1);
        assert_eq!(
            dropped.id_to_instance[1], LIGHT_ID_NONE,
            "a's id is retired"
        );
        assert_eq!(dropped.instance_to_id[0], 2, "b keeps its id");

        // Re-add 'a'. It must take a fresh id, not reclaim the retired 1.
        let readded = registry.reconcile(&[light("a", "m", 0), light("b", "m", 1)], 2);
        let a_id = readded.instance_to_id[0];
        assert_ne!(a_id, 1, "a retired id must never be reused");
        assert_eq!(readded.instance_to_id[1], 2, "b is untouched throughout");
        assert_eq!(
            readded.id_to_instance[1], LIGHT_ID_NONE,
            "1 stays a tombstone"
        );
    }

    /// T2: a mesh edit reshuffles triangle indices, so a stored
    /// `(primitive, barycentrics)` reference is no longer valid — the
    /// fingerprint change drops the history (fresh id), while a material or
    /// transform edit (same fingerprint) keeps it.
    #[test]
    fn a_mesh_change_drops_history_but_a_material_change_keeps_it() {
        let mut registry = LightIdentityRegistry::new();
        registry.reconcile(&[light("lamp", "mesh_v1", 0)], 1);

        // Same fingerprint (only the material changed): id survives.
        let kept = registry.reconcile(&[light("lamp", "mesh_v1", 0)], 1);
        assert_eq!(kept.instance_to_id[0], 1, "a material edit keeps the id");

        // Fingerprint changed (the mesh was edited): fresh id, old retired.
        let dropped = registry.reconcile(&[light("lamp", "mesh_v2", 0)], 1);
        let new_id = dropped.instance_to_id[0];
        assert_ne!(new_id, 1, "a mesh edit drops the old id");
        assert_eq!(dropped.id_to_instance[1], LIGHT_ID_NONE, "old id retired");
        assert_eq!(dropped.id_to_instance[new_id as usize], 0);
    }
}
