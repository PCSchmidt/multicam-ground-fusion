# multicam-ground-fusion: Evidence Notes

Companion to multicam-ground-fusion.md. Labels: Confirmed, Ran it, Per the docs, My reading, Ask the author, Author says. Paths are repository-relative. No secrets were read or printed; no .env exists in the tracked files.

## E1 Fact sheet

| Fact | Value | Label |
| --- | --- | --- |
| Name, licence | multicam-ground-fusion, MIT (LICENSE file; pyproject "MIT") | Confirmed |
| Kind, status | research prototype and measured study, v0.4.0 (pyproject, README) | Confirmed |
| Languages, size | Python only. src 623 lines (7 files), tests 263 lines (4 files), scripts 1,156 lines (11 Python files, one shell script) | Confirmed |
| Commits | 18, 2026-10-09 to 2026-10-14; all 18 carry one identical AI co-author trailer | Confirmed |
| Tests | 22 collected, 22 passed in a scratch copy (17 s) | Ran it |
| Clean copy | Scratch copy from `git archive HEAD`; dependencies installed into a scratch folder (scipy, pytest, hypothesis; numpy was present) | Ran it |
| Demo | `examples/demo.py` output identical to README (7.8 m/s, IMPLAUSIBLE_SPEED on track 1) | Ran it |
| Synthetic benchmark | `scripts/benchmark.py` reproduces the README/RESULTS.md table exactly | Ran it |
| Needs | CPU only for tests and benchmark; GPU, PyTorch and a 6.8 GB dataset for the real-data studies (not run) | Per the docs |
| CI | One job: pip install, `mypy src tests`, `pytest -q` on Ubuntu, Python 3.12. No ruff step, no benchmark step | Confirmed |
| Deployed | No. Dockerfile only runs `pytest -q`. No hosting, no live link | Confirmed |
| Users | None evidenced | Ask the author |
| Related | README Motivation: step beyond sitework-ai (SiteWatch AI), same homography idea, same "detector proposes, gates decide" philosophy | Per the docs |

## E2 Reading and running log

Read: README, all six docs (three protocols, four results files including RESULTS.md), tracker.py, geometry.py, mot.py, sim.py, tests/test_fusion.py, pyproject, Dockerfile, ci.yml, .gitignore, demo, detect.py, finetune_detector.py, git log with times, key result JSON structures. Skimmed: the other scripts, mmptrack.py, wildtrack.py, the other tests.
Run (all in a scratch copy, outside the clone): pytest, demo, benchmark, a probe script (E4), a check of the chi-square gate value (9.2103). Not run: study scripts (large downloads, GPU), mypy, ruff.
Clone state: not written to. `git status` shows no changes to tracked files. Test caches were disabled.

## E3 Claims check (level L1 = doc matches committed result file unless stated)

| ID | Claim | Verdict |
| --- | --- | --- |
| CL-01 | 22 tests pass | Supported (L2) |
| CL-02 | Synthetic table, 10 held-out seeds, two cameras 0.93 to 0.96, one 0.61 to 0.67 | Supported (L2, benchmark re-run matches) |
| CL-03 | Wildtrack oracle fuse-7 MOTA 0.793 [0.740, 0.844], IDF1 0.63, 145 switches | Supported (L1, results/real_test.json) |
| CL-04 | Wildtrack COCO detector fuse-7 -0.169 [-0.252, -0.068], 4,555 FP, 723 FN, 257 switches | Supported (L1) |
| CL-05 | Live-only variants 0.917 and 0.102 | Supported (L1); labelled exploratory by the docs |
| CL-06 | Single camera medians 0.582 (oracle) and 0.006 (detector); best 0.761 and 0.241 | Supported (L1, recomputed from the per-camera entries) |
| CL-07 | No-gate oracle: 0.68 MOTA (-0.11), switches 145 to 393 | Supported (L1) |
| CL-08 | MMPTrack: oracle 0.997, 7 switches, 41,632 person-frames; detector A 0.568, B 0.589, C 0.603 | Supported (L1 against README and RESULTS_SECOND_SCENE.md; second_test.json structure not recomputed) |
| CL-09 | Detector study pooled MOTA B: D0 0.524, D1 0.594, D2 0.759, O 0.986, with intervals | Supported (L1, detectors_test.json pooled and bootstrap entries, D0 and D2 differences) |
| CL-10 | Fine-tuning +0.235 [0.164, 0.304] over D0; D1 +0.070 | Supported (L1) |
| CL-11 | Five of seven written hypotheses met | Supported (L1, table in RESULTS_DETECTORS.md) |
| CL-12 | "ruff" listed among the checks in the README at-a-glance | Partly: ruff is a dev dependency, CI does not run it |
| CL-13 | Protocols were written before results | Partly: commit order supports it (E8); push times unavailable |
| CL-14 | Flags "never silently corrected" | Supported by code reading and test; no code path removes a flag |

## E4 Rule attacks and probes (scratch copy, Ran it)

- Box far from track: spawned a new track (also covered by an existing test).
- Two identical boxes, same camera, same step: tracks 2, and the second, ghost track is confirmed after further steps. Not handled.
- Unknown camera name: KeyError. Not handled.
- NaN box: accepted without error, produced a tentative track. Not followed further.
- dt = 0 and dt = -0.1: accepted without error.
- Horizon point: ValueError aborts the whole step (apply_homography raises; step() does not catch).
- Fewer than 4 calibration pairs: ValueError.
- Four collinear calibration points: returns a numerically degenerate matrix (entries around 4e15), no error.
- Empty detection list on a fresh tracker: returns an empty list.
- Flag removal: no code path removes IMPLAUSIBLE_SPEED.

## E5 Code map

- src/multicam_fusion: geometry.py (homography, DLT via SVD, bottom_center, covariance_ellipse), tracker.py (config, Track, GroundPlaneTracker: predict, mahalanobis_sq, update, spawn, step, confirmed), mot.py (CLEAR-MOT and IDF1, sticky matching), sim.py (seeded scenario), wildtrack.py and mmptrack.py (readers).
- scripts: benchmark.py; detect.py, detect2.py, detect_mmp.py (detector passes, cached); finetune_detector.py; run_real.py, run_second.py, run_mmp.py, run_detectors.py (study runners; test stage one-shot and refuses to overwrite); extract_wildtrack.py (recovers a zip with bad offsets, CRC check); fetch_mmp2.sh.
- Entry points: examples/demo.py, scripts/*.py. No package CLI.

## E6 Model call sites

| Where | Model | Purpose | Settings |
| --- | --- | --- | --- |
| scripts/detect.py | torchvision fasterrcnn_resnet50_fpn_v2, COCO_V1 weights | person boxes on Wildtrack, 1920x1080, cached | cache floor score 0.3; study threshold 0.9 |
| scripts/detect2.py, detect_mmp.py | Faster R-CNN v2 (D0), RetinaNet v2 (D1), fine-tuned (D2) | person boxes on MMPTrack, 640x360 | thresholds D0 0.9, D1 0.7, D2 0.9 (results/detectors_dev.json per docs) |
| scripts/finetune_detector.py | Faster R-CNN v2, new 2-class head | fine-tuning on MMPTrack session 63am, every 15th frame, 2 epochs, SGD lr 0.01, momentum 0.9, wd 1e-4, batch 4, cosine decay, flip; seeds random.seed(0), torch.manual_seed(0) | Confirmed |

No paid services, no API keys, no prompts. Weights downloaded from the torchvision hosting at run time; licence of the weights not checked (Ask the author).

## E7 Tests and CI

22 tests: test_fusion.py (10 incl. a hypothesis property test, 15 examples, and a 5-seed simulation test), test_mot.py, test_tracker_options.py, test_wildtrack.py. Spot checks: (1) test_far_detection_is_gated_not_merged asserts 2 tracks after a box 27 m away appears, real behaviour; (2) test_implausible_speed_is_flagged_not_hidden uses a lowered ceiling (1 m/s) and a loosened gate to force the flag, so it tests the mechanism, not realism; (3) test_fusion_beats_single_camera_in_overlap_heavy_scene averages 5 simulated seeds at dropout 0.4, a weak test of a coverage effect (the docs say coverage dominates). Coverage not measured. CI runs mypy and pytest only. The README claim that mypy is strict and clean was corrected in the last commit; not re-run here.

## E8 History and plan-versus-results

| Study | Plan commit | Results commit |
| --- | --- | --- |
| Wildtrack | b33e597, 2026-10-12 15:45 | c26fbfb, 2026-10-12 16:29 (runner e7b0e25 at 16:26, deviations logged there) |
| MMPTrack | ad086c7, 2026-10-13 07:20 | 76b3d1e, 2026-10-13 08:15 (runner 835c435 at 08:13) |
| Detectors | 5144e97, 2026-10-13 16:56 (scripts 59a092c one minute later, thresholds f1afda8 at 20:52) | 17ae7df, 2026-10-14 12:42 |

Plans precede results by commit time. Push times were not available, and the Wildtrack runner and results commits are three minutes apart, so the order of the test run against the deviation notes rests on the docs' statement. Bulk commits: 76b3d1e (1,997 insertions), 17ae7df (2,356), e7b0e25 (3,919). Co-author credit: 18 of 18 commits, one identical trailer; the tool is not named in the guide, per the author's standing instruction.

## E9 Gates and approvals

No gate files or approval records. The one-shot runners are the closest thing (refuse to overwrite test result files, per scripts).

## E10 Environment

pyproject lists minimum versions only (numpy>=1.26, scipy>=1.12; dev extras; study extras torch>=2.4, torchvision>=0.19). No lock file. Results files do not record library versions. Scratch run used Python 3.14 with current scipy, hypothesis and pytest; tests passed, and the benchmark table reproduced exactly.

## E11 Security and privacy

No keys, no outside services except weight downloads. A scan of tracked files for key, token, secret and password terms found no matches outside the docs (ran a code-only search). The data are person-tracking footage; none is committed (data/ is git-ignored). Dataset copies come from unofficial mirrors with no licence text found. Dependabot is enabled (monthly, Actions and pip). GitHub security settings not checked.

## E12 Shortcomings register (top items)

| # | Shortcoming | Severity | Disclosed | Options |
| --- | --- | --- | --- | --- |
| 1 | One fixed measurement noise for every box | Medium | Yes (Limitations) | A: per-box uncertainty from the homography's Jacobian (medium effort, likely high payoff); B: noise growing with range by a fitted rule; C: leave and document |
| 2 | No appearance cue, crossing people can swap | Medium | Yes | A: add a learned appearance model (breaks the all-fixed design); B: leave and document |
| 3 | Speed flag ineffective at low frame rates | Medium | Yes (never fired) | A: jump-versus-uncertainty flag; B: frame-rate-scaled ceiling |
| 4 | Duplicate identical boxes create a ghost track | Low | No | A: suppress near-duplicate boxes before matching; B: leave |
| 5 | A bad box aborts a whole step (horizon error, unknown camera) | Low | Partly | A: skip and count bad boxes; B: leave |
| 6 | No version lock, no recorded library versions | Low | No | Pin or record versions with results |
| 7 | CI does not run ruff or any evaluation | Low | README implies ruff | Add ruff and a benchmark threshold step |
| 8 | Thresholds at grid edge (0.9) in the detector study | Low | Yes | Extend the grid and rerun on development frames |

## E13 Method detail

Gate: chi2.ppf(0.99, df=2) = 9.2103 (Ran it). Kalman: state [x, y, vx, vy], F with dt, Q from acceleration noise via G, R = sigma^2 I, H selects position. Toy calculations in the guide were checked by hand: gain 0.36/0.61 = 0.590, estimate 10.531, variance 0.1475; Mahalanobis squared 2.25 and 11.11; Hungarian toy totals 11 versus 4; MOTA toy 0.80; real MOTA (723+4555+257)/4733 = 1.1695, so -0.169. IDF1 = 2 IDTP/(n_gt + n_pred). Reference: Bernardin and Stiefelhagen 2008 (CLEAR-MOT, named in mot.py). Complexity: matching is cubic in the number of tracks and boxes per camera step, cost matrix quadratic.

## E14 Discrepancies

README at-a-glance and Verify sections cite ruff; CI does not run it. README says "10 held-out seeds" and benchmark seeds 100-109, consistent. Detector study D0 config A scored 0.485 on fresh sequences against 0.568 earlier (different sequences; the docs flag it). Protocol table lists office_0 in both fine-tuning and development rows, on different sessions (per the docs).

## E15 Production readiness

Rung: works locally. Checklist: clean-start run Partly (tests yes, studies need data and GPU); CI blocks bad change Partly (tests and types, no evaluation); quality measured against baselines Yes; safe on failure Partly; human-in-the-loop Not applicable; versioning Partly (version number, no lock); logs and monitoring No; data and secrets Yes (none committed, no keys). Gap list: service wrapper, live video, per-site calibration, monitoring, retraining plan, runbook, security review.

## E16 Review dispositions

Not yet reviewed by an independent pass.

## E17 Completeness and backlog

Done and tested: homography, DLT, tracker, gate, flags, scoring, simulator, readers, studies. Done, untested: study scripts and detector code. Not built (README): YOLO plus ByteTrack, extended Kalman filter, PETS 2009, bird's-eye-view renderer, reprojection-error drift flag. Backlog: Fix (ghost-track suppression, bad-box handling); Improve (per-box uncertainty, a frame-rate-aware flag); Add (benchmark gate in CI, runbook, structured logging, a third site). Next milestone: per-box uncertainty, exit test on Wildtrack development frames only.

## Diagram sources

Images were drawn with a plotting script (matplotlib) in a scratch folder; the three pictures show the fixed/learned split, the parts, and the headline results (values from E3).

---

## Revision log (v1 draft to current)

The previous text is kept as `multicam-ground-fusion.v1draft.md`. All ten fixes in `multicam-ground-fusion.readercheck.md` were applied. Word count by plain `wc -w`: v1 5,336; current 4,598 (target 4,300 to 4,600; cap 8,000).

| Fix | What changed |
| --- | --- |
| 1. Shorter opening | Opening is now short paragraphs with four numbers (0.99, 0.52, 0.76, minus 0.17); 0.79, 0.93 to 0.96 and the rest moved to the results chapter. |
| 2. MOTA | Defined with a 100-moment toy example (20 errors, 0.80) in the opening, including why a score can go negative. |
| 3. Dataset names | Wildtrack and MMPTrack each defined in plain words at first mention. |
| 4. Speed ceiling | Checked the repo: the tracker default is 12 m/s (code and README), the demo lowers it to 8 (demo script, README), and 4 m/s is an exploratory setting in the Wildtrack results. Text now says it is a setting and names each value; the parts picture says "12 m/s by default". Demo output (7.8 m/s, IMPLAUSIBLE_SPEED) matches the README. |
| 5. Methods trimmed | Seven techniques cut to the idea of each in one to three sentences; DLT, SVD, Mahalanobis, chi-square and the full Kalman arithmetic removed from the body. |
| 6. Development vs test, off-the-shelf vs fine-tuned | Defined once, early (opening and detectors chapter). |
| 7. COCO, ResNet50, FPN | 46.7 and 41.5, ResNet50, FPN and the thresholds dropped; COCO defined in one clause; "two standard detectors plus one retrained". |
| 8. Stricter rule | Explained plainly in the results chapter (original: three updates, possibly one moment, coasting tracks reported for up to five moments; stricter: three different moments, live tracks only). Checked against the tracker's configuration. Itemised the seven predictions: E1 to E4 and E7 met, E5 and E6 not. |
| 9. Glossary | Grew from 6 to 24 entries (recall, seed, threshold, identity switch, bootstrap, development and test data, and others). |
| 10. Softened claims | "Fusion helps most when the detector is weak" became "helped when the detector was weak but not consistently" (RESULTS_DETECTORS: fusion beat the best single camera in five of six weak-detector cases, not on Wildtrack with the real detector, and lost in two of three rooms with the fine-tuned one); "well-measured" and "sound evaluation" replaced by "careful evaluation on small data"; "plans came first" kept only as "the commit order shows", with the unprovable push times stated once. Data paragraph and "Likely questions" block split. |

Also removed: the edge-case table (replaced by a short paragraph), the SiteWatch AI folder name and the defense-interest sentence, the "unofficial copies" caveat is stated once in the data chapter and once in the opening. Pictures regenerated with a matplotlib script kept outside the repo and viewed after rendering (no clipped text): the parts picture (speed ceiling wording), the results picture ("COCO" replaced by "off-the-shelf", clearer footnote; bar values unchanged: 0.793, minus 0.169, 0.524, 0.594, 0.759, 0.986), and the fixed-versus-learned picture (Kalman filter wording made plain).

Not re-verified: no study was re-run; numbers are from the committed result files.
