// Test-only kernel: reports the byte sizes of Slang structs the host
// mirrors with hand-counted constants (`SHADOW_RAY_SIZE` in wavefront.rs
// — no Rust mirror struct pins it, since the host never reads a record),
// so growing one side without the other fails a test instead of silently
// overrunning a queue. Compiled at test time through the hot-reload
// compiler; never embedded, never shipped (`slangc.rs` lists what ships).

import pathstate;

struct Params {
    uint* sizes;
};

[vk::push_constant]
Params params;

[shader("compute")]
[numthreads(1, 1, 1)]
void layout_test(uint3 tid: SV_DispatchThreadID) {
    params.sizes[0] = sizeof(ShadowRay);
    params.sizes[1] = sizeof(QueueState);
}
