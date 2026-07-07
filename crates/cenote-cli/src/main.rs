//! Headless batch renderer. Grows a real interface (`--width/--height/--out/--watch`)
//! when there is a renderer behind it (m0-plan §4, step 5).

fn main() {
    println!(
        "cenote-cli {} — M0 scaffold, no renderer yet",
        env!("CARGO_PKG_VERSION")
    );
}
