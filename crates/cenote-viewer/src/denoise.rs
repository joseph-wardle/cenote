//! The denoised view: a worker thread runs the published frame through
//! OIDN about once a second, and the last result waits on the GPU for the
//! tonemap to read in place of the raw average.
//!
//! Denoising is a *view* of the film, never part of the estimator — the
//! render thread accumulates unchanged underneath, and flipping the
//! toggle changes only which buffer the tonemap reads. One job is in
//! flight at a time: the filter costs ~200 ms of CPU at 720p, and a
//! preview that trails the film by a second is what the cadence promises.
//! The trade shows during motion — the denoised image lags the orbit by
//! up to that second — which is the usual shape of an interactive
//! denoise toggle.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context as _;

/// At most one denoise per second: frequent enough to track convergence,
/// rare enough that the CPU filter never contends with the display loop.
const CADENCE: Duration = Duration::from_secs(1);

/// One frame's inputs, downloaded from the publish buffers.
struct Job {
    width: u32,
    height: u32,
    beauty: Vec<f32>,
    albedo: Vec<f32>,
    normal: Vec<f32>,
}

/// The worker's answer: the filtered beauty and the size it was filtered at.
struct Filtered {
    width: u32,
    height: u32,
    beauty: Vec<f32>,
}

/// The filtered beauty, resident again, and the size it belongs to.
struct Display {
    buffer: cenote::gpu::Buffer,
    width: u32,
    height: u32,
}

/// The main-thread half: submits at the cadence, keeps the latest result
/// uploaded. The worker half is the thread spawned in [`DenoiseView::new`];
/// it exits when this half drops the job channel.
pub struct DenoiseView {
    jobs: mpsc::Sender<Job>,
    results: mpsc::Receiver<cenote::Result<Filtered>>,
    display: Option<Display>,
    /// A job is with the worker; the cadence tick skips until it answers.
    busy: bool,
    /// When the in-flight or last job was submitted.
    submitted: Option<Instant>,
    /// The worker reported a filter failure (a broken OIDN install, most
    /// likely every time) — logged once, raw frames from then on.
    dead: bool,
}

impl DenoiseView {
    /// Spawn the worker. The OIDN device starts on the first job, so an
    /// untouched toggle costs one idle thread and nothing else.
    pub fn new() -> Self {
        let (jobs, worker_jobs) = mpsc::channel::<Job>();
        let (worker_results, results) = mpsc::channel();
        thread::spawn(move || {
            let mut denoiser = None;
            while let Ok(job) = worker_jobs.recv() {
                if worker_results.send(filter(&mut denoiser, &job)).is_err() {
                    return; // the viewer is gone
                }
            }
        });
        Self {
            jobs,
            results,
            display: None,
            busy: false,
            submitted: None,
            dead: false,
        }
    }

    /// Pump the view: land a finished result on the GPU, and when the
    /// worker is idle and the cadence has elapsed, ship it the current
    /// frame. Call once per redraw while the toggle is on.
    pub fn update(
        &mut self,
        gpu: &cenote::gpu::Context,
        frame: &cenote::render::Frame,
    ) -> anyhow::Result<()> {
        if self.dead {
            return Ok(());
        }
        while let Ok(result) = self.results.try_recv() {
            self.busy = false;
            match result {
                Ok(filtered) => {
                    let buffer =
                        cenote::render::Tonemap::upload_average(gpu, "denoised", &filtered.beauty)
                            .context("uploading the denoised frame")?;
                    self.display = Some(Display {
                        buffer,
                        width: filtered.width,
                        height: filtered.height,
                    });
                }
                Err(error) => {
                    log::error!("denoise failed — showing raw frames from here on: {error}");
                    self.dead = true;
                    return Ok(());
                }
            }
        }
        let due = self.submitted.is_none_or(|at| at.elapsed() >= CADENCE);
        if !self.busy && due {
            let job = Job {
                width: frame.width(),
                height: frame.height(),
                beauty: texels(&gpu.download_buffer(frame.image())?),
                albedo: texels(&gpu.download_buffer(frame.albedo())?),
                normal: texels(&gpu.download_buffer(frame.normal())?),
            };
            // The worker owns its end until we drop ours; send can't fail.
            let _ = self.jobs.send(job);
            self.busy = true;
            self.submitted = Some(Instant::now());
        }
        Ok(())
    }

    /// The buffer the tonemap should read for `frame`: the denoised
    /// beauty, unless none matching the frame's size exists yet (the
    /// first second after enabling, or right after a resize).
    pub fn display(&self, frame: &cenote::render::Frame) -> Option<&cenote::gpu::Buffer> {
        self.display
            .as_ref()
            .filter(|display| (display.width, display.height) == (frame.width(), frame.height()))
            .map(|display| &display.buffer)
    }
}

/// The worker's whole job: an OIDN device on first use (≈40 ms, paid on
/// the first toggle, not at startup), then one balanced-quality filter.
fn filter(denoiser: &mut Option<cenote::denoise::Denoiser>, job: &Job) -> cenote::Result<Filtered> {
    let denoiser = match denoiser {
        Some(denoiser) => denoiser,
        None => denoiser.insert(cenote::denoise::Denoiser::new()?),
    };
    let beauty = denoiser.denoise(
        job.width,
        job.height,
        cenote::denoise::Quality::Balanced,
        &job.beauty,
        &job.albedo,
        &job.normal,
    )?;
    Ok(Filtered {
        width: job.width,
        height: job.height,
        beauty,
    })
}

/// Reinterpret downloaded bytes as the f32 texels they are.
fn texels(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|lanes| f32::from_ne_bytes([lanes[0], lanes[1], lanes[2], lanes[3]]))
        .collect()
}
